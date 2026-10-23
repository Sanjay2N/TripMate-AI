"""TripMate UI: profile and trip form on the left, plan and chat in the middle.

Run with:  uv run streamlit run app/streamlit_app.py
"""

import streamlit as st

from tripmate.agent.orchestrator import Agent
from tripmate.agent.whatif import what_if
from tripmate.config import get_settings
from tripmate.llm.client import serving_llm
from tripmate.profile import TravelProfile, extract_profile, load_profile, save_profile
from tripmate.rag.retriever import Retriever
from tripmate.schemas import Mode, Pace, Party, Tier, TripRequest
from tripmate.state import experience_score

EMERGENCY = "All-India emergency 112 · Ambulance 108 · Tourist helpline 1363 · Women's helpline 1091"
INTERESTS = [
    "beaches", "history", "temples", "food", "nature", "wildlife", "adventure",
    "shopping", "art and museums", "nightlife", "trekking", "architecture",
]  # fmt: skip

st.set_page_config(page_title="TripMate AI", page_icon="🧭", layout="wide")
settings = get_settings()
PROFILE_DIR = settings.data_dir / "profiles"


@st.cache_resource
def get_agent() -> Agent:
    return Agent(Retriever(settings), serving_llm(settings), settings)


agent = get_agent()
ss = st.session_state
ss.setdefault("profile", load_profile("me", PROFILE_DIR) or TravelProfile())
ss.setdefault("state", None)
ss.setdefault("last", None)
ss.setdefault("sim", None)
profile: TravelProfile = ss.profile


def run_turn(message: str | None = None) -> None:
    with st.spinner("TripMate is planning… a small local model takes a little while."):
        ss.last = agent.turn(ss.state, message) if message else agent.turn(ss.state)
    ss.sim = None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧭 TripMate AI")
    st.caption(f"Model: `{settings.llm_model}`")

    with st.expander("My travel profile", expanded=False):
        profile.home_city = st.selectbox(
            "Home city", sorted(agent.places), index=sorted(agent.places).index(profile.home_city)
        )
        about = st.text_area(
            "Tell me about your past trips and tastes",
            value=profile.notes,
            placeholder="We loved slow mornings in Coorg, hate crowded temples, all vegetarian…",
        )
        if st.button("Learn from this") and about.strip():
            facts = extract_profile(agent.llm, about)
            if facts:
                profile.notes, profile.pace = about, facts.pace
                profile.interests = list(dict.fromkeys(profile.interests + facts.interests))
                profile.learn(facts.constraints)
                st.success("Profile updated from your description.")
            else:
                st.warning("Could not read that; nothing was changed.")
        st.write("**Pace:**", profile.pace.value)
        st.write("**Interests:**", ", ".join(profile.interests) or "none yet")
        st.write("**Always respect:**", "; ".join(profile.constraints) or "nothing yet")
        if st.button("Save profile"):
            save_profile(profile, PROFILE_DIR)
            st.success("Saved.")
        st.download_button(
            "Download profile", profile.model_dump_json(indent=2), "tripmate_profile.json"
        )
        uploaded = st.file_uploader("Load a profile", type="json")
        if uploaded:
            ss.profile = TravelProfile.model_validate_json(uploaded.read())
            st.rerun()

    st.subheader("New trip")
    cities = agent.retriever.destinations
    with st.form("trip"):
        sources = sorted(agent.places)
        source = st.selectbox("From", sources, index=sources.index(profile.home_city))
        destinations = st.multiselect("To (one or more cities)", cities, default=["Goa"])
        days = st.number_input("Days", 1, 14, 4)
        col1, col2 = st.columns(2)
        travelers = col1.number_input("Travelers", 1, 12, profile.travelers)
        party = col2.selectbox("Party", list(Party), index=list(Party).index(profile.party))
        mode = col1.selectbox("Travel by", list(Mode), index=list(Mode).index(profile.mode))
        pace = col2.selectbox("Pace", list(Pace), index=list(Pace).index(profile.pace))
        budget = st.number_input("Total budget (₹)", 2000, 2_000_000, 40_000, step=1000)
        tier = st.radio(
            "Plan variant",
            list(Tier),
            index=list(Tier).index(profile.tier),
            horizontal=True,
            help="Budget uses up to 60% of your ceiling, Balanced 80%, Premium 100%.",
        )
        interests = st.multiselect(
            "Interests", sorted(set(INTERESTS + profile.interests)), default=profile.interests
        )
        constraints = st.text_input(
            "Must respect (separate with ;)", value="; ".join(profile.constraints)
        )
        submitted = st.form_submit_button("Plan my trip", type="primary")

    if submitted:
        if not destinations:
            st.error("Pick at least one destination.")
        elif source in destinations:
            st.error("Your home city cannot also be a destination.")
        elif days < len(destinations):
            st.error(f"{len(destinations)} cities need at least {len(destinations)} days.")
        else:
            request = TripRequest(
                source=source,
                destinations=destinations,
                days=days,
                travelers=travelers,
                party=party,
                mode=mode,
                budget_inr=budget,
                tier=tier,
                pace=pace,
                interests=interests,
                constraints=[c.strip() for c in constraints.split(";") if c.strip()],
            )
            with st.spinner("Finding places…"):
                ss.state = agent.start(request)
            run_turn()

# ---------------------------------------------------------------- main
state = ss.state
if state is None:
    st.header("Plan a trip")
    st.write(
        "Fill in the trip form on the left. TripMate builds a day-by-day plan from "
        "Wikivoyage listings, keeps it inside your budget, and lets you refine it by chat."
    )
    st.stop()

req = state.request
route = " → ".join([req.source] + [f"{s.city} ({len(s.days)}d)" for s in state.route] + [req.source])
st.header(route)

left, right = st.columns([3, 2], gap="large")

with left:
    last = ss.last
    if last and last.violations:
        st.warning(
            "This plan still breaks some checks, so treat it as a draft:\n\n- "
            + "\n- ".join(last.violations[:6])
        )
    if last and not last.applied and last.reply and state.plan and not last.violations:
        st.info("No change was made to the plan this turn.")

    if state.plan:
        for d in state.plan.days:
            with st.expander(f"Day {d.day} · {state.day_city(d.day)} · {d.theme}", expanded=True):
                for a in d.activities:
                    p = state.pois.get(a.poi, {})
                    st.markdown(
                        f"**{a.start}** · **{p.get('name', a.poi)}** · {a.duration_min} min · "
                        f"₹{a.cost_inr}/person  \n{a.why}"
                    )
                    facts = [
                        f"Hours: {p.get('hours') or 'unknown'}",
                        f"Listed price: {p.get('price') or 'unknown'}",
                        f"Verified: {p.get('lastedit') or 'date unknown (low confidence)'}",
                    ]
                    st.caption(" · ".join(facts) + f" · [source]({p.get('url', '')})")
                drop = state.pois.get(d.drop_if_short, {}).get("name")
                add = state.pois.get(d.add_if_extra, {}).get("name")
                if drop or add:
                    st.markdown(
                        f"↘︎ *Short on time?* Drop **{drop or '—'}**  ·  "
                        f"↗︎ *Extra energy?* Add **{add or '—'}**"
                    )

    st.subheader("Chat")
    for m in state.history:
        st.chat_message(m["role"]).write(m["content"])
    if last and last.changes:
        st.caption("What changed: " + " · ".join(last.changes))
    if last and last.ask:
        st.caption(f"TripMate asks: {last.ask}")
    if message := st.chat_input("e.g. Make day 2 more relaxed and add a museum"):
        run_turn(message)
        st.rerun()

with right:
    b = agent.budget(state)
    if b:
        st.subheader("Budget")
        c1, c2, c3 = st.columns(3)
        c1.metric("Planned", f"₹{b.total:,}")
        c2.metric(f"{req.tier.value.title()} target", f"₹{b.target:,}")
        c3.metric("Your ceiling", f"₹{b.ceiling:,}")
        st.progress(min(b.total / b.ceiling, 1.0))
        st.table(
            {
                "Item": ["Intercity transport*", "Stay", "Food", "Local transport", "Activities"],
                "₹": [b.intercity, b.stay, b.food, b.local_transport, b.activities],
            }
        )
        st.caption("*Distance-based estimate, not a fare quote (low confidence).")
        for city, c in state.plan.cities.items():
            if c.stay:
                st.markdown(
                    f"**Stay in {city}:** {c.stay.area}, ₹{c.stay.per_night_inr:,}/night per room  \n"
                    f"{c.stay.why}"
                )
        score = experience_score(state)
        if score is not None:
            st.metric(
                "Experience score",
                f"{score}/100",
                help="70% share of your interests covered by at least one activity, "
                "30% share of activities that match an interest.",
            )

    st.subheader("Heads-up")
    if not state.heads_up:
        st.caption("Nothing flagged yet.")
    for h in state.heads_up:
        src = state.guides.get(h.source) or state.pois.get(h.source) or {}
        st.markdown(f"⚠️ **{h.risk}**  \n{h.mitigation}")
        unverified = " · unverified, machine-compiled dataset" if src.get("verified") is False else ""
        st.caption(f"Based on [{src.get('area', h.source)}]({src.get('url', '')}){unverified}")
    st.caption(EMERGENCY)

    if req.constraints:
        st.subheader("I'm keeping in mind")
        st.write("; ".join(req.constraints))
        new = [c for c in req.constraints if c.lower() not in {x.lower() for x in profile.constraints}]
        if new and st.button("Remember these for future trips"):
            profile.learn(new)
            save_profile(profile, PROFILE_DIR)
            st.rerun()

    if state.plan:
        st.subheader("What if…")
        with st.form("whatif"):
            pct = st.slider("Budget change (%)", -40, 40, 0, step=10)
            delta = st.slider("Days", -2, 2, 0)
            new_pace = st.select_slider("Pace", list(Pace), value=req.pace)
            if st.form_submit_button("Simulate"):
                try:
                    with st.spinner("Simulating…"):
                        ss.sim = what_if(agent, state, pct, delta, new_pace)
                except ValueError as e:
                    st.error(str(e))
        sim = ss.sim
        if sim:
            st.write(sim.summary)
            sb = agent.budget(sim.state)
            if sb:
                st.caption(
                    f"Simulated total ₹{sb.total:,} of ₹{sb.target:,}"
                    + ("" if sim.used_model else " · answered instantly, no model call")
                )
            for change in sim.changes:
                st.markdown(f"- {change}")
            if st.button("Apply this version"):
                ss.state, ss.sim, ss.last = sim.state, None, sim.result
                st.rerun()

st.divider()
st.caption(
    "Place data from [Wikivoyage](https://en.wikivoyage.org) (CC BY-SA 4.0). Hours and prices "
    "may be out of date; check the 'Verified' date on each place. Seasonal, safety and typical "
    "price notes also draw on the Kaggle Indian Tourism Dataset (Apache-2.0), which is unverified."
)
