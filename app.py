"""
NBA Win Predictor - Streamlit Demo App

One-time terminal setup:
    pip install "transformers==4.44.2"

Run with:
    streamlit run app.py
"""

import json
import math
import random
import re
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import pipeline

st.set_page_config(page_title="NBA Win Predictor", layout="wide")

MAX_LEN = 100
PROB_FLOOR = 0.01
PROB_CAP = 0.99
QUARTER_LEN_SEC = 12 * 60
TOTAL_GAME_SEC = 48 * 60
SCORE_TIME_K = 0.035
TIME_FRAC_FLOOR = 0.0005
SENTIMENT_K = 0.6

DEMO_HOME_LABEL = "San Antonio Spurs"
DEMO_AWAY_LABEL = "New York Knicks"


def get_demo_checkpoint_table():
    rows = [
        {"Game Clock": "Q1 9:00", "Home Score": 8, "Away Score": 6, "Commentary": ""},
        {"Game Clock": "Q1 3:00", "Home Score": 18, "Away Score": 14,
         "Commentary": "Victor Wembanyama is putting on a show for San Antonio, already up to 12 points and dominating the paint early."},
        {"Game Clock": "Q1 0:00", "Home Score": 26, "Away Score": 24, "Commentary": ""},
        {"Game Clock": "Q2 6:00", "Home Score": 36, "Away Score": 40,
         "Commentary": "Karl-Anthony Towns is heating up for New York, already up to 16 points and controlling the paint for the Knicks."},
        {"Game Clock": "Q2 0:00", "Home Score": 48, "Away Score": 50, "Commentary": ""},
        {"Game Clock": "Q3 9:00", "Home Score": 54, "Away Score": 54,
         "Commentary": "Breaking: Victor Wembanyama appears to be injured and is heading to the locker room for evaluation - a massive blow for the Spurs."},
        {"Game Clock": "Q3 3:00", "Home Score": 58, "Away Score": 64,
         "Commentary": "The backup center is overwhelmed without Wembanyama, and San Antonio's offense completely stalls."},
        {"Game Clock": "Q3 0:00", "Home Score": 64, "Away Score": 70, "Commentary": ""},
        {"Game Clock": "Q4 6:00", "Home Score": 78, "Away Score": 76,
         "Commentary": "San Antonio mounts a furious rally, cutting the deficit and energizing the home crowd."},
        {"Game Clock": "Q4 3:00", "Home Score": 84, "Away Score": 86,
         "Commentary": "Jalen Brunson takes over for New York in crunch time, already up to 28 points and refusing to let the Knicks lose this one."},
        {"Game Clock": "Q4 0:00", "Home Score": 90, "Away Score": 96, "Commentary": ""},
    ]
    return pd.DataFrame(rows)


def market_calibrate(p):
    return min(max(p, PROB_FLOOR), PROB_CAP)


def parse_game_clock(clock_str):
    match = re.match(r"\s*(Q[1-4]|OT)\s+(\d+):(\d+)\s*", str(clock_str).strip(), re.IGNORECASE)
    if not match:
        return 1, 12, 0
    q_str, mm, ss = match.groups()
    quarter = 5 if q_str.upper() == "OT" else int(q_str[1])
    return quarter, int(mm), int(ss)


def format_game_clock(quarter, mm, ss):
    q_str = "OT" if quarter == 5 else f"Q{quarter}"
    return f"{q_str} {mm}:{ss:02d}"


def to_elapsed_seconds(quarter, mm, ss):
    return (quarter - 1) * QUARTER_LEN_SEC + (QUARTER_LEN_SEC - (mm * 60 + ss))


def from_elapsed_seconds(elapsed_sec):
    elapsed_sec = max(0, elapsed_sec)
    if elapsed_sec > 0 and elapsed_sec % QUARTER_LEN_SEC == 0:
        quarter = int(elapsed_sec // QUARTER_LEN_SEC)
        return quarter, 0, 0
    quarter = int(elapsed_sec // QUARTER_LEN_SEC) + 1
    into_quarter_sec = elapsed_sec % QUARTER_LEN_SEC
    remaining_sec = QUARTER_LEN_SEC - into_quarter_sec
    mm = int(remaining_sec // 60)
    ss = int(remaining_sec % 60)
    return quarter, mm, ss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=1,
                 dim_feedforward=256, dropout=0.1, max_len=256, pretrained_embeddings=None):
        super().__init__()
        self.d_model = d_model
        if pretrained_embeddings is not None:
            self.token_embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, token_ids, attention_mask=None):
        x = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            text_repr = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            text_repr = encoded.mean(dim=1)
        return text_repr


class StatsEncoder(nn.Module):
    def __init__(self, n_numeric_features, hidden_dim=32, out_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_numeric_features, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )

    def forward(self, numeric_features):
        return self.net(numeric_features)


class DualBranchFusionModel(nn.Module):
    def __init__(self, vocab_size, n_numeric_features, text_dim=128, stats_dim=32,
                 n_heads=4, n_text_layers=1, classifier_hidden=64, dropout=0.1, max_len=256,
                 pretrained_embeddings=None):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, text_dim, n_heads, n_text_layers, dropout=dropout,
                                         max_len=max_len, pretrained_embeddings=pretrained_embeddings)
        self.stats_encoder = StatsEncoder(n_numeric_features, out_dim=stats_dim, dropout=dropout)
        fusion_dim = text_dim + stats_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, token_ids, numeric_features, attention_mask=None):
        text_repr = self.text_encoder(token_ids, attention_mask)
        stats_repr = self.stats_encoder(numeric_features)
        fused = torch.cat([text_repr, stats_repr], dim=1)
        return self.classifier(fused).squeeze(-1)


def simple_tokenize(text):
    return re.findall(r"[a-z0-9]+|[.,%]", text.lower())


def encode_text(text, vocab, max_len=MAX_LEN):
    tokens = simple_tokenize(text)
    ids = [vocab.get(t, 1) for t in tokens][:max_len]
    attention_mask = [1] * len(ids)
    pad_len = max_len - len(ids)
    ids = ids + [0] * pad_len
    attention_mask = attention_mask + [0] * pad_len
    return ids, attention_mask


NUMERIC_COLS = [
    "home_prev5_win_pct", "home_prev5_avg_points",
    "home_prev10_win_pct", "home_prev10_avg_points",
    "home_top_scorer_value", "home_top_playmaker_value", "home_top_rebounder_value",
    "away_prev5_win_pct", "away_prev5_avg_points",
    "away_prev10_win_pct", "away_prev10_avg_points",
    "away_top_scorer_value", "away_top_playmaker_value", "away_top_rebounder_value",
]


def generate_narrative(home_team, away_team, home_win_pct, home_scorer, home_playmaker, home_rebounder,
                        away_team_win_pct, away_scorer, away_playmaker, away_rebounder):
    favorite = home_team if home_win_pct >= away_team_win_pct else away_team
    templates = [
        (
            f"Tonight's matchup pits {home_team} against {away_team}, and the numbers favor {favorite} heading in. "
            f"{home_team} have won {home_win_pct * 100:.0f} percent of their last five games, led by {home_scorer} scoring "
            f"and {home_playmaker} setting the table. {away_team} have gone {away_team_win_pct * 100:.0f} percent over their last five, "
            f"relying on {away_scorer} for buckets and {away_rebounder} on the glass."
        ),
        (
            f"{home_team} host {away_team} in a matchup with plenty of storylines. {home_team} enter at {home_win_pct * 100:.0f} percent "
            f"over their last five, with {home_rebounder} controlling the boards and {home_scorer} leading the scoring column. "
            f"{away_team} counter at {away_team_win_pct * 100:.0f} percent, powered by {away_playmaker} distributing and {away_scorer} finishing."
        ),
        (
            f"All eyes are on {home_team} versus {away_team} tonight. {home_team} bring a {home_win_pct * 100:.0f} percent mark over their "
            f"last five games, anchored by {home_scorer} and {home_playmaker}. {away_team} arrive at {away_team_win_pct * 100:.0f} percent, "
            f"with {away_rebounder} dominating inside and {away_scorer} carrying the scoring load."
        ),
        (
            f"{home_team} and {away_team} square off in what should be a competitive night. Over their last five games, "
            f"{home_team} sit at {home_win_pct * 100:.0f} percent behind {home_scorer}, while {away_team} check in at {away_team_win_pct * 100:.0f} percent "
            f"thanks to strong two-way play from {away_playmaker} and {away_rebounder}."
        ),
    ]
    return random.choice(templates)


@st.cache_resource
def load_artifacts():
    with open("vocab.json") as f:
        vocab = json.load(f)
    with open("standardization_stats.json") as f:
        stats = json.load(f)
    means = pd.Series(stats["means"])
    stds = pd.Series(stats["stds"])

    with open("calibration.json") as f:
        raw_temperature = json.load(f)["temperature"]
    temperature = max(abs(raw_temperature), 0.5)

    model = DualBranchFusionModel(vocab_size=len(vocab), n_numeric_features=len(NUMERIC_COLS), text_dim=100)
    model.load_state_dict(torch.load("trained_model.pt", map_location="cpu"))
    model.eval()

    teams = pd.read_csv("latest_team_stats.csv")
    return vocab, means, stds, model, teams, temperature


@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


vocab, train_means, train_stds, model, teams_df, TEMPERATURE = load_artifacts()
sentiment_pipeline = load_sentiment_model()
teams_df["label"] = teams_df["teamCity"].astype(str) + " " + teams_df["teamName"].astype(str)


def raw_logit(text, numeric_raw):
    numeric_arr = (np.array(numeric_raw, dtype="float32") - train_means[NUMERIC_COLS].values) / train_stds[NUMERIC_COLS].values
    ids, mask = encode_text(text, vocab)
    token_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.tensor([mask], dtype=torch.long)
    numeric_features = torch.tensor(np.array([numeric_arr]), dtype=torch.float32)
    with torch.no_grad():
        logit = model(token_ids, numeric_features, attention_mask).item()
    return logit


def score_time_adjustment(home_score, away_score, elapsed_sec):
    time_remaining_frac = max(TIME_FRAC_FLOOR, min(1.0, (TOTAL_GAME_SEC - elapsed_sec) / TOTAL_GAME_SEC))
    margin = home_score - away_score
    return SCORE_TIME_K * margin / math.sqrt(time_remaining_frac)


def commentary_sentiment_adjustment(commentary_text, home_city, home_name, away_city, away_name):
    if not commentary_text:
        return 0.0
    result = sentiment_pipeline(commentary_text[:512])[0]
    polarity = 1.0 if result["label"] == "POSITIVE" else -1.0
    confidence = result["score"]
    text_lower = commentary_text.lower()
    mentions_home = (home_city.lower() in text_lower) or (home_name.lower() in text_lower) or ("home team" in text_lower)
    mentions_away = (away_city.lower() in text_lower) or (away_name.lower() in text_lower) or ("away team" in text_lower)
    if mentions_home and not mentions_away:
        team_sign = 1.0
    elif mentions_away and not mentions_home:
        team_sign = -1.0
    else:
        team_sign = 0.0
    return SENTIMENT_K * polarity * team_sign * confidence


def team_card(col, city, name, prob, is_favorite):
    border_color = "#16a34a" if is_favorite else "#6b7280"
    bg_color = "rgba(22,163,74,0.08)" if is_favorite else "rgba(107,114,128,0.06)"
    with col:
        st.markdown(
            f"""
            <div style="border: 2px solid {border_color}; border-radius: 12px; padding: 20px;
                        background-color: {bg_color}; text-align: center;">
                <div style="font-size: 16px; color: #6b7280; font-weight: 600;">{city}</div>
                <div style="font-size: 22px; font-weight: 700; margin-bottom: 8px;">{name}</div>
                <div style="font-size: 42px; font-weight: 800; color: {border_color};">{prob * 100:.1f}%</div>
                <div style="font-size: 13px; color: #6b7280;">chance to win</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prob)


def render_prob_chart(placeholder, results_rows, home_name, away_name):
    df = pd.DataFrame(results_rows)
    df["away_win_prob"] = 1 - df["home_win_prob"]
    df["order"] = range(len(df))
    long_df = df.melt(id_vars=["label", "order"], value_vars=["home_win_prob", "away_win_prob"],
                       var_name="side", value_name="prob")
    long_df["Team"] = long_df["side"].map({"home_win_prob": home_name, "away_win_prob": away_name})

    zoom_x = alt.selection_interval(bind="scales", encodings=["x"])

    chart = alt.Chart(long_df).mark_line(point=True).encode(
        x=alt.X("order:Q", title="Game progress", axis=alt.Axis(labels=False)),
        y=alt.Y("prob:Q", title="Win probability", scale=alt.Scale(domain=[0, 1], clamp=True), axis=alt.Axis(format="%")),
        color=alt.Color("Team:N", legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("Team:N"), alt.Tooltip("prob:Q", format=".1%", title="Win %"), alt.Tooltip("label:N", title="Time")],
    ).properties(height=320).add_params(zoom_x)

    placeholder.altair_chart(chart, use_container_width=True)


def render_feed_table(placeholder, results_rows, home_name):
    partial_df = pd.DataFrame(results_rows)
    display_df = partial_df[partial_df["commentary"] != ""][["label", "home_score", "away_score", "commentary", "home_win_prob"]].copy()
    display_df["home_win_prob"] = (display_df["home_win_prob"] * 100).round(1).astype(str) + "%"
    display_df.columns = ["Time", "Home", "Away", "Commentary", f"{home_name} win %"]
    display_df = display_df.iloc[::-1].reset_index(drop=True)
    placeholder.dataframe(display_df, use_container_width=True, hide_index=True, height=420)


def build_box_score(results_df, home_name, away_name):
    box = {"Team": [home_name, away_name]}
    quarter_labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "OT"}
    prev_home, prev_away = 0, 0
    for q in [1, 2, 3, 4, 5]:
        q_rows = results_df[results_df["quarter"] == q]
        if q_rows.empty:
            continue
        end_home = q_rows["home_score"].iloc[-1]
        end_away = q_rows["away_score"].iloc[-1]
        box[quarter_labels[q]] = [end_home - prev_home, end_away - prev_away]
        prev_home, prev_away = end_home, end_away
    box["Final"] = [prev_home, prev_away]
    return pd.DataFrame(box)


SCORING_EVENT_POINTS = [
    ("{player} hits a three-pointer for {team}!", 3),
    ("{player} drills a step-back three.", 3),
    ("{player} converts an and-one at the rim.", 3),
    ("{player} completes a three-point play at the line.", 3),
    ("{player} scores on a fast break for {team}.", 2),
    ("{player} knocks down a mid-range jumper.", 2),
    ("{player} gets a putback off the offensive rebound.", 2),
    ("{player} finishes a nice give-and-go for {team}.", 2),
    ("{player} sets up a teammate for an easy bucket.", 2),
    ("{player}'s no-look pass leads to a fast-break score.", 2),
    ("{player} sinks a free throw for {team}.", 1),
]
DEFENSIVE_EVENTS = [
    "{player} comes up with a big steal for {team}.", "{player} blocks the shot at the rim.",
    "{player} draws an offensive foul.",
]
NEUTRAL_EVENTS = [
    "{team} calls a timeout to reset.", "{team} misses badly from deep.",
    "The referees review a call at the scorer's table.", "{team} misses the front end of the free throws.",
]

INJURY_EVENTS = [
    "Breaking: {player} is dealing with an apparent injury and heads to the locker room for evaluation.",
    "{player} is shaken up after a hard fall and is being looked at by the training staff.",
    "{player} grabs at his ankle after an awkward landing and is limping noticeably.",
]
FOUL_TROUBLE_EVENTS = [
    "{player} picks up a technical foul after arguing a call.",
    "{player} is battling foul trouble and may see limited minutes down the stretch.",
    "{player} picks up his fourth foul and heads to the bench as a precaution.",
]
MOMENTUM_EVENTS = [
    "{team} calls a timeout after a costly turnover swings momentum.",
    "{team} goes on an emphatic run, energizing the home crowd.",
    "{team} answers right back with a run of their own to shift the energy.",
    "{team} looks rattled after a string of unforced turnovers.",
]
HOT_STREAK_EVENTS = [
    "{player} is heating up for {team}, already up to {pts} points and looking to keep it going.",
    "{player}, who leads {team} in scoring, has poured in {pts} points already tonight.",
    "{player} is feeling it from deep tonight - {pts} points and counting for {team}.",
    "{player} continues to carry {team} offensively with {pts} points on the night.",
    "{player} is putting on a show for {team}, up to {pts} points so far.",
    "{player} can't miss right now - {pts} points for {team} and the crowd is buzzing.",
]
ALL_STORYLINE_EVENTS = INJURY_EVENTS + FOUL_TROUBLE_EVENTS + MOMENTUM_EVENTS + HOT_STREAK_EVENTS


def decompose_points(diff):
    parts = []
    remaining = diff
    while remaining > 0:
        if remaining in (1, 2, 3):
            parts.append(remaining)
            remaining = 0
        else:
            step = random.choice([2, 3])
            parts.append(step)
            remaining -= step
    return parts


def jittered_fractions(n):
    weights = [random.uniform(0.4, 1.6) for _ in range(n)]
    total = sum(weights)
    cum = 0.0
    fractions = []
    for w in weights:
        cum += w
        fractions.append(cum / total)
    fractions[-1] = 1.0
    return fractions


def build_interval_events(start_home, start_away, end_home, end_away, start_elapsed, end_elapsed, home, away, ejected):
    home_diff = max(0, end_home - start_home)
    away_diff = max(0, end_away - start_away)
    home_parts = decompose_points(home_diff)
    away_parts = decompose_points(away_diff)

    def pick_player(side):
        candidates = [c for c in [side["scorer"], side["playmaker"], side["rebounder"]] if c not in ejected]
        return random.choice(candidates) if candidates else "a bench player"

    events = []
    for pts in home_parts:
        matching = [t for t, p in SCORING_EVENT_POINTS if p == pts]
        template = random.choice(matching)
        player = pick_player(home)
        events.append({"delta_home": pts, "delta_away": 0, "flavor": template.format(player=player, team=home["name"])})
    for pts in away_parts:
        matching = [t for t, p in SCORING_EVENT_POINTS if p == pts]
        template = random.choice(matching)
        player = pick_player(away)
        events.append({"delta_home": 0, "delta_away": pts, "flavor": template.format(player=player, team=away["name"])})

    n_flavor = random.randint(4, 8)
    for _ in range(n_flavor):
        side = home if random.random() < 0.5 else away
        if random.random() < 0.5:
            player = pick_player(side)
            text = random.choice(DEFENSIVE_EVENTS)
            flavor = text.format(player=player, team=side["name"])
        else:
            text = random.choice(NEUTRAL_EVENTS)
            flavor = text.format(team=side["name"])
        events.append({"delta_home": 0, "delta_away": 0, "flavor": flavor})

    random.shuffle(events)

    n = len(events)
    fractions = jittered_fractions(n)
    cum_home, cum_away = start_home, start_away
    result = []
    for i, ev in enumerate(events):
        elapsed = start_elapsed + (end_elapsed - start_elapsed) * fractions[i]
        cum_home += ev["delta_home"]
        cum_away += ev["delta_away"]
        result.append({"elapsed_sec": elapsed, "home_score": cum_home, "away_score": cum_away, "flavor": ev["flavor"]})
    return result


def simulate_realistic_game(home_avg_pts, away_avg_pts, home_pregame_prob, home, away, seed):
    rng = np.random.RandomState(seed)
    home_bias = (home_pregame_prob - 0.5) * 8
    home_total = max(75, int(rng.normal(home_avg_pts + home_bias, 9)))
    away_total = max(75, int(rng.normal(away_avg_pts - home_bias, 9)))

    home_shares = rng.dirichlet([5, 5, 5, 5])
    away_shares = rng.dirichlet([5, 5, 5, 5])
    home_q_pts = [round(home_total * s) for s in home_shares]
    away_q_pts = [round(away_total * s) for s in away_shares]
    home_q_pts[-1] += home_total - sum(home_q_pts)
    away_q_pts[-1] += away_total - sum(away_q_pts)

    rows = []
    cum_home, cum_away = 0, 0
    for q in range(1, 5):
        mid_home = cum_home + round(home_q_pts[q - 1] * 0.5)
        mid_away = cum_away + round(away_q_pts[q - 1] * 0.5)
        rows.append({"Game Clock": format_game_clock(q, 6, 0), "Home Score": mid_home, "Away Score": mid_away, "Commentary": ""})
        cum_home += home_q_pts[q - 1]
        cum_away += away_q_pts[q - 1]
        rows.append({"Game Clock": format_game_clock(q, 0, 0), "Home Score": cum_home, "Away Score": cum_away, "Commentary": ""})

    n_events = max(5, len(rows) // 2)
    candidate_idxs = list(range(1, len(rows) - 1))
    event_idxs = sorted(rng.choice(candidate_idxs, size=min(n_events, len(candidate_idxs)), replace=False))

    tech_count = {}
    ejected_players = set()

    for idx in event_idxs:
        acting_home = rng.random() < 0.5
        player = home["scorer"] if acting_home else away["scorer"]
        team_name = home["name"] if acting_home else away["name"]
        team_score_so_far = rows[idx]["Home Score"] if acting_home else rows[idx]["Away Score"]

        if player in ejected_players:
            safe_events = [e for e in ALL_STORYLINE_EVENTS if "technical" not in e and "ejected" not in e]
            text = rng.choice(safe_events)
        else:
            text = rng.choice(ALL_STORYLINE_EVENTS)
            if "technical foul" in text:
                tech_count[player] = tech_count.get(player, 0) + 1
                if tech_count[player] >= 2:
                    text = "{player} is ejected from the game after picking up a second technical foul!"
                    ejected_players.add(player)

        if "{pts}" in text:
            pts = max(2, round(team_score_so_far * rng.uniform(0.25, 0.4)))
            rows[idx]["Commentary"] = text.format(player=player, team=team_name, pts=pts)
        else:
            rows[idx]["Commentary"] = text.format(player=player, team=team_name)

    return pd.DataFrame(rows)


st.title("🏀 NBA Win Market")
st.caption("Stage 1: lock in pregame odds. Stage 2: watch a stats-grounded, randomized game unfold live.")

team_labels = teams_df["label"].tolist()

demo_col1, demo_col2 = st.columns([3, 1])
with demo_col2:
    if st.button("🎬 Load Demo Script (Spurs vs Knicks)", use_container_width=True):
        if DEMO_HOME_LABEL in team_labels and DEMO_AWAY_LABEL in team_labels:
            st.session_state["home_team"] = DEMO_HOME_LABEL
            st.session_state["away_team"] = DEMO_AWAY_LABEL
            st.session_state["demo_mode"] = True
            st.session_state.pop("pregame_locked", None)
            st.session_state.pop("sim_results", None)
            st.session_state.pop("preview", None)
            st.session_state.pop("narrative_text", None)
            st.session_state.pop("narrative_teams", None)
            st.rerun()
        else:
            st.error("Spurs/Knicks not found in latest_team_stats.csv - check team names in your data.")

col1, col2 = st.columns(2)
with col1:
    home_label = st.selectbox("Home team", team_labels, index=0, key="home_team")
with col2:
    default_away_idx = 1 if len(team_labels) > 1 else 0
    away_label = st.selectbox("Away team", team_labels, index=default_away_idx, key="away_team")

if home_label != DEMO_HOME_LABEL or away_label != DEMO_AWAY_LABEL:
    st.session_state["demo_mode"] = False

if home_label == away_label:
    st.warning("Pick two different teams.")
    st.stop()

home_row = teams_df[teams_df["label"] == home_label].iloc[0]
away_row = teams_df[teams_df["label"] == away_label].iloc[0]

home_people = {
    "name": home_row["teamName"], "scorer": f"{home_row['top_scorer_firstName']} {home_row['top_scorer_lastName']}",
    "playmaker": f"{home_row['top_playmaker_firstName']} {home_row['top_playmaker_lastName']}",
    "rebounder": f"{home_row['top_rebounder_firstName']} {home_row['top_rebounder_lastName']}",
}
away_people = {
    "name": away_row["teamName"], "scorer": f"{away_row['top_scorer_firstName']} {away_row['top_scorer_lastName']}",
    "playmaker": f"{away_row['top_playmaker_firstName']} {away_row['top_playmaker_lastName']}",
    "rebounder": f"{away_row['top_rebounder_firstName']} {away_row['top_rebounder_lastName']}",
}

numeric_raw = [
    home_row["prev5_win_pct"], home_row["prev5_avg_points"],
    home_row["prev10_win_pct"], home_row["prev10_avg_points"],
    home_row["top_scorer_value"], home_row["top_playmaker_value"], home_row["top_rebounder_value"],
    away_row["prev5_win_pct"], away_row["prev5_avg_points"],
    away_row["prev10_win_pct"], away_row["prev10_avg_points"],
    away_row["top_scorer_value"], away_row["top_playmaker_value"], away_row["top_rebounder_value"],
]

baseline_text = generate_narrative(
    f"{home_row['teamCity']} {home_row['teamName']}", f"{away_row['teamCity']} {away_row['teamName']}",
    home_row["prev5_win_pct"], home_people["scorer"], home_people["playmaker"], home_people["rebounder"],
    away_row["prev5_win_pct"], away_people["scorer"], away_people["playmaker"], away_people["rebounder"],
)

if st.session_state.get("narrative_teams") != (home_label, away_label):
    st.session_state["narrative_text"] = baseline_text
    st.session_state["narrative_teams"] = (home_label, away_label)
    st.session_state.pop("preview", None)

if st.session_state.get("locked_matchup") != (home_label, away_label):
    st.session_state.pregame_locked = False
    st.session_state.pop("sim_results", None)
    st.session_state.pop("game_seed", None)

# =============================================================
# STAGE 1: Pregame Analysis
# =============================================================
st.header("Stage 1 — Pregame Analysis")

if st.session_state.get("demo_mode"):
    st.info("🎬 Demo mode active: Spurs vs Knicks, scripted checkpoint table pre-loaded below.")

if not st.session_state.get("pregame_locked", False):
    st.subheader("Pregame expert narrative (auto-generated, editable)")
    text_input = st.text_area("Narrative text", height=140, key="narrative_text")

    if st.button("Preview pregame odds", use_container_width=True):
        logit = raw_logit(text_input, numeric_raw)
        prob = market_calibrate(1 / (1 + math.exp(-logit / TEMPERATURE)))
        st.session_state.preview = prob

    if "preview" in st.session_state:
        prob = st.session_state.preview
        c1, c2 = st.columns(2)
        team_card(c1, home_row["teamCity"], home_row["teamName"], prob, prob >= 1 - prob)
        team_card(c2, away_row["teamCity"], away_row["teamName"], 1 - prob, 1 - prob > prob)

        if st.button("🔒 Lock in pregame odds and start the game", type="primary", use_container_width=True):
            st.session_state.pregame_locked = True
            st.session_state.locked_matchup = (home_label, away_label)
            st.session_state.pregame_text = text_input
            st.session_state.pregame_prob = prob
            st.rerun()
else:
    prob = st.session_state.pregame_prob
    st.success("Pregame odds locked in.")
    c1, c2 = st.columns(2)
    team_card(c1, home_row["teamCity"], home_row["teamName"], prob, prob >= 1 - prob)
    team_card(c2, away_row["teamCity"], away_row["teamName"], 1 - prob, 1 - prob > prob)
    with st.expander("Pregame narrative used"):
        st.write(st.session_state.pregame_text)
    if st.button("Reset pregame (start over)"):
        st.session_state.pregame_locked = False
        st.session_state.pop("sim_results", None)
        st.session_state.pop("game_seed", None)
        st.rerun()

# =============================================================
# STAGE 2: Live Game
# =============================================================
if st.session_state.get("pregame_locked", False):
    st.header("Stage 2 — Live Game")
    st.caption(
        "This section simulates the game: enter/edit key moments below (score, game clock, optional news), "
        "then click Run. The engine fills in realistic plays between your checkpoints and the odds update live "
        "based on YOUR typed commentary - a real pretrained language model reads it."
    )

    if "game_seed" not in st.session_state:
        st.session_state.game_seed = random.randint(0, 1_000_000)

    with st.expander("Step 1: Game checkpoints (edit freely)", expanded="sim_results" not in st.session_state):
        st.caption(
            "**Game Clock** format: `Q1 10:05` (quarter + minutes:seconds remaining, `OT 5:00` for overtime). "
            "Scores are cumulative totals at that moment. **Commentary** is optional and reserved for real storylines - "
            "these actually move the odds."
        )

        if st.session_state.get("demo_mode"):
            default_rows = get_demo_checkpoint_table()
            editor_key = "live_events_editor_demo"
            if st.button("🎲 Randomize instead (exit demo script)"):
                st.session_state["demo_mode"] = False
                st.session_state.game_seed = random.randint(0, 1_000_000)
                st.session_state.pop("sim_results", None)
                st.rerun()
        else:
            if st.button("🎲 Randomize game"):
                st.session_state.game_seed = random.randint(0, 1_000_000)
                st.session_state.pop("sim_results", None)
            default_rows = simulate_realistic_game(
                home_row["prev10_avg_points"], away_row["prev10_avg_points"],
                st.session_state.pregame_prob, home_people, away_people, st.session_state.game_seed,
            )
            editor_key = f"live_events_editor_{st.session_state.game_seed}"

        edited = st.data_editor(
            default_rows,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Game Clock": st.column_config.TextColumn(help="Format: Q1 10:05, Q4 0:00, OT 3:15"),
                "Home Score": st.column_config.NumberColumn(min_value=0, step=1),
                "Away Score": st.column_config.NumberColumn(min_value=0, step=1),
                "Commentary": st.column_config.TextColumn(width="large", help="Optional real storylines that move the odds"),
            },
            key=editor_key,
        )
        run_clicked = st.button("▶️ Run live game", type="primary", use_container_width=True)

    if run_clicked:
        cumulative_text = st.session_state.pregame_text
        cumulative_sentiment_adj = 0.0
        ejected_players_live = set()
        rows = [{
            "label": "Pregame", "quarter": 0, "home_score": 0, "away_score": 0,
            "commentary": "(locked pregame odds)", "home_win_prob": st.session_state.pregame_prob,
        }]

        st.markdown("### Step 2: Live feed")
        chart_placeholder = st.empty()
        cards_placeholder = st.empty()
        table_placeholder = st.empty()
        status_placeholder = st.empty()

        prev_home, prev_away = 0, 0
        prev_elapsed = 0

        for _, r in edited.iterrows():
            quarter, mm, ss = parse_game_clock(r["Game Clock"])
            current_elapsed = to_elapsed_seconds(quarter, mm, ss)
            home_score = int(r["Home Score"])
            away_score = int(r["Away Score"])
            commentary = str(r["Commentary"]).strip() if pd.notna(r["Commentary"]) else ""

            interval_events = build_interval_events(prev_home, prev_away, home_score, away_score,
                                                      prev_elapsed, current_elapsed, home_people, away_people,
                                                      ejected_players_live)

            for f in interval_events:
                f_q, f_mm, f_ss = from_elapsed_seconds(f["elapsed_sec"])
                f_label = format_game_clock(f_q, f_mm, f_ss)
                base_logit = raw_logit(cumulative_text, numeric_raw)
                adj = score_time_adjustment(f["home_score"], f["away_score"], f["elapsed_sec"])
                combined_logit = base_logit + adj + cumulative_sentiment_adj
                filler_prob = market_calibrate(1 / (1 + math.exp(-combined_logit / TEMPERATURE)))

                rows.append({
                    "label": f_label, "quarter": f_q, "home_score": f["home_score"], "away_score": f["away_score"],
                    "commentary": f["flavor"], "home_win_prob": filler_prob,
                })
                render_prob_chart(chart_placeholder, rows, home_row["teamName"], away_row["teamName"])
                status_placeholder.caption(f"⏱️ {f_label} — {f['flavor']}")
                render_feed_table(table_placeholder, rows, home_row["teamName"])
                time.sleep(0.5)

            label = format_game_clock(quarter, mm, ss)
            status_placeholder.info(f"⏱️ {label}")

            if commentary:
                cumulative_text = cumulative_text + " " + commentary
                cumulative_sentiment_adj += commentary_sentiment_adjustment(
                    commentary, home_row["teamCity"], home_row["teamName"],
                    away_row["teamCity"], away_row["teamName"],
                )
                if "ejected" in commentary.lower():
                    for candidate_name in [home_people["scorer"], home_people["playmaker"], home_people["rebounder"],
                                            away_people["scorer"], away_people["playmaker"], away_people["rebounder"]]:
                        if candidate_name.lower() in commentary.lower():
                            ejected_players_live.add(candidate_name)

            base_logit = raw_logit(cumulative_text, numeric_raw)
            adj = score_time_adjustment(home_score, away_score, current_elapsed)
            combined_logit = base_logit + adj + cumulative_sentiment_adj
            live_prob = market_calibrate(1 / (1 + math.exp(-combined_logit / TEMPERATURE)))

            fallback_flavor = interval_events[-1]["flavor"] if interval_events else ""
            final_commentary = commentary if commentary else fallback_flavor
            is_true_end = (quarter == 4 and current_elapsed == TOTAL_GAME_SEC)
            if is_true_end:
                label = "Game Ends"
                final_commentary = f"Game Ends - Final: {home_row['teamName']} {home_score}, {away_row['teamName']} {away_score}."

            rows.append({
                "label": label, "quarter": quarter, "home_score": home_score, "away_score": away_score,
                "commentary": final_commentary, "home_win_prob": live_prob,
            })

            render_prob_chart(chart_placeholder, rows, home_row["teamName"], away_row["teamName"])

            with cards_placeholder.container():
                c1, c2 = st.columns(2)
                team_card(c1, home_row["teamCity"], home_row["teamName"], live_prob, live_prob >= 1 - live_prob)
                team_card(c2, away_row["teamCity"], away_row["teamName"], 1 - live_prob, 1 - live_prob > live_prob)

            render_feed_table(table_placeholder, rows, home_row["teamName"])

            prev_home, prev_away = home_score, away_score
            prev_elapsed = current_elapsed
            time.sleep(4.5 if commentary else 1.5)

        status_placeholder.success("Game complete.")
        st.session_state.sim_results = rows

    if "sim_results" in st.session_state:
        if not run_clicked:
            st.markdown("### Live feed (from last run)")
            render_feed_table(st.empty(), st.session_state.sim_results, home_row["teamName"])

        results_df = pd.DataFrame(st.session_state.sim_results)
        st.markdown("### Box score")
        box_df = build_box_score(results_df, home_row["teamName"], away_row["teamName"])
        st.dataframe(box_df, use_container_width=True, hide_index=True)
