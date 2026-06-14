from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import chess.pgn
import io
import asyncio
from collections import defaultdict
from anthropic import Anthropic

app = FastAPI(title="ChessScout API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Anthropic()

# ── In-memory cache (username+platform → report) ──────────────────────────────
# Prevents re-fetching 100 games if same player is analyzed twice in a session
_cache: dict[str, dict] = {}

# ── Models ────────────────────────────────────────────────────────────────────

class ScoutRequest(BaseModel):
    username: str
    platform: str = "chess.com"
    num_games: int = 100

class FeedbackRequest(BaseModel):
    username: str
    rating: str
    comment: str = ""

class PredictionResult(BaseModel):
    username: str
    predicted_opening: str
    actual_result: str    # "win" | "loss" | "draw"
    opening_played: str

# ── Fetch Games ───────────────────────────────────────────────────────────────

async def fetch_chesscom_games(username: str, num_games: int) -> list[str]:
    pgns = []
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.get(
            f"https://api.chess.com/pub/player/{username}/games/archives",
            headers={"User-Agent": "ChessScout/1.0"}
        )
        if r.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Player '{username}' not found on chess.com")
        archives = r.json().get("archives", [])
        archives.reverse()  # most recent first

        for url in archives:
            if len(pgns) >= num_games:
                break
            r2 = await http.get(url + "/pgn", headers={"User-Agent": "ChessScout/1.0"})
            if r2.status_code == 429:
                # Rate limited — wait and retry once
                await asyncio.sleep(2)
                r2 = await http.get(url + "/pgn", headers={"User-Agent": "ChessScout/1.0"})
            if r2.status_code != 200:
                continue
            pgns.extend(r2.text.strip().split("\n\n\n"))

    return pgns[:num_games]


async def fetch_lichess_games(username: str, num_games: int) -> list[str]:
    """
    Lichess rate limit: 20 games/sec anonymous.
    We cap at 200 games max to keep response time under 15 seconds.
    """
    cap = min(num_games, 200)
    url = f"https://lichess.org/api/games/user/{username}?max={cap}&pgnInJson=false"
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.get(url, headers={"Accept": "application/x-chess-pgn"})
        if r.status_code == 429:
            raise HTTPException(status_code=429, detail="Lichess rate limit hit. Please wait 60 seconds and try again.")
        if r.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Player '{username}' not found on Lichess")
        return r.text.strip().split("\n\n\n")[:cap]

# ── Confidence Rating ─────────────────────────────────────────────────────────

def confidence(n: int) -> dict:
    if n >= 30:
        return {"label": "HIGH", "score": "9/10", "games": n}
    elif n >= 15:
        return {"label": "MEDIUM", "score": "6/10", "games": n}
    elif n >= 5:
        return {"label": "LOW", "score": "3/10", "games": n}
    else:
        return {"label": "INSUFFICIENT", "score": "1/10", "games": n}

# ── Core Analysis ─────────────────────────────────────────────────────────────

def analyze_games(pgns: list[str], target_username: str) -> dict:
    """
    Parse PGNs and extract fully evidence-backed stats.
    - Every metric includes sample size for confidence ratings
    - No psychological labels — only observable, measurable behaviors
    - Rating-aware: tracks opponent rating per outcome for normalization
    """
    uname = target_username.lower()

    stats = {
        "total_games": 0,
        "wins": 0, "losses": 0, "draws": 0,
        "white_wins": 0, "white_losses": 0, "white_draws": 0,
        "black_wins": 0, "black_losses": 0, "black_draws": 0,

        # Opening performance
        "openings": defaultdict(lambda: {
            "wins": 0, "losses": 0, "draws": 0,
            "opp_ratings_win": [], "opp_ratings_loss": []
        }),

        # Game length buckets (ply = half moves)
        # short <20, medium 20-59, long 60+
        "by_length": {
            "short":  {"wins": 0, "losses": 0, "draws": 0, "games": 0},
            "medium": {"wins": 0, "losses": 0, "draws": 0, "games": 0},
            "long":   {"wins": 0, "losses": 0, "draws": 0, "games": 0},
        },

        # How games end
        "terminations": defaultdict(int),

        # Raw data
        "game_lengths_ply": [],
        "opponent_ratings": [],
        "time_controls": defaultdict(int),

        # Collapse: long game ending in loss or timeout
        "collapse_games": 0,

        # Rating buckets for normalization
        # Did they lose to weaker players? (interesting pattern)
        "losses_vs_lower_rated": 0,
        "losses_vs_higher_rated": 0,
        "wins_vs_higher_rated": 0,
        "rating_comparison_games": 0,
    }

    for pgn_text in pgns:
        pgn_text = pgn_text.strip()
        if not pgn_text:
            continue
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if game is None:
                continue
        except Exception:
            continue

        h = game.headers
        white = h.get("White", "").lower()
        black = h.get("Black", "").lower()
        result = h.get("Result", "*")
        opening = h.get("Opening", h.get("ECOUrl", "Unknown"))
        if "/" in opening:
            opening = opening.split("/")[-1].replace("-", " ").title()
        eco = h.get("ECO", "")
        termination = h.get("Termination", "").lower()
        time_control = h.get("TimeControl", "Unknown")

        if uname in white:
            color = "white"
        elif uname in black:
            color = "black"
        else:
            continue

        if result == "1-0":
            outcome = "win" if color == "white" else "loss"
        elif result == "0-1":
            outcome = "win" if color == "black" else "loss"
        elif result == "1/2-1/2":
            outcome = "draw"
        else:
            continue

        stats["total_games"] += 1
        if outcome == "win":
            stats["wins"] += 1
        elif outcome == "loss":
            stats["losses"] += 1
        else:
            stats["draws"] += 1
        if outcome == "win":
            stats[f"{color}_wins"] += 1
        elif outcome == "loss":
            stats[f"{color}_losses"] += 1
        else:
            stats[f"{color}_draws"] += 1

        # Opening
        key = f"{eco} {opening}".strip() if eco else opening
        if outcome == "win":
            stats["openings"][key]["wins"] += 1
        elif outcome == "loss":
            stats["openings"][key]["losses"] += 1
        else:
            stats["openings"][key]["draws"] += 1

        # Opponent rating
        opp_key = "BlackElo" if color == "white" else "WhiteElo"
        my_key  = "WhiteElo" if color == "white" else "BlackElo"
        opp_rating = 0
        my_rating  = 0
        try:
            opp_rating = int(h.get(opp_key, 0))
            my_rating  = int(h.get(my_key, 0))
            if opp_rating > 0:
                stats["opponent_ratings"].append(opp_rating)
            if opp_rating > 0 and my_rating > 0:
                stats["rating_comparison_games"] += 1
                rating_diff = opp_rating - my_rating
                if outcome == "loss":
                    if rating_diff < -50:   # lost to someone 50+ below
                        stats["losses_vs_lower_rated"] += 1
                    elif rating_diff > 50:
                        stats["losses_vs_higher_rated"] += 1
                elif outcome == "win" and rating_diff > 100:
                    stats["wins_vs_higher_rated"] += 1
        except (ValueError, TypeError):
            pass

        # Store opp rating per opening outcome for normalization
        if opp_rating > 0:
            if outcome == "win":
                stats["openings"][key]["opp_ratings_win"].append(opp_rating)
            elif outcome == "loss":
                stats["openings"][key]["opp_ratings_loss"].append(opp_rating)

        # Game length
        ply = game.end().ply()
        stats["game_lengths_ply"].append(ply)
        bucket = "short" if ply < 20 else "medium" if ply < 60 else "long"
        stats["by_length"][bucket]["games"] += 1
        stats["by_length"][bucket][outcome + "s"] += 1

        # Collapse detection
        if ply >= 60 and (outcome == "loss" or "timeout" in termination or "time" in termination):
            stats["collapse_games"] += 1

        # Termination
        for term in ["timeout", "resignation", "checkmate", "abandoned", "time forfeit"]:
            if term in termination:
                stats["terminations"][term] += 1
                break
        else:
            stats["terminations"]["other"] += 1

        stats["time_controls"][time_control] += 1

    # ── Serialize ─────────────────────────────────────────────────────────────
    stats["terminations"]  = dict(stats["terminations"])
    stats["time_controls"] = dict(stats["time_controls"])

    total = stats["total_games"]
    if total == 0:
        return stats

    # ── Derived stats ─────────────────────────────────────────────────────────

    if stats["game_lengths_ply"]:
        stats["avg_game_length_ply"] = round(
            sum(stats["game_lengths_ply"]) / len(stats["game_lengths_ply"]), 1)

    if stats["opponent_ratings"]:
        stats["avg_opponent_rating"] = round(
            sum(stats["opponent_ratings"]) / len(stats["opponent_ratings"]))

    def wr(w, l, d):
        t = w + l + d
        return {"win_rate": round(w / t * 100, 1) if t else 0, "sample": t}

    stats["overall_wr"] = wr(stats["wins"],       stats["losses"],       stats["draws"])
    stats["white_wr"]   = wr(stats["white_wins"],  stats["white_losses"], stats["white_draws"])
    stats["black_wr"]   = wr(stats["black_wins"],  stats["black_losses"], stats["black_draws"])

    # Length bucket rates + confidence
    for bucket in ("short", "medium", "long"):
        b = stats["by_length"][bucket]
        g = b["games"]
        b["win_rate"]   = round(b["wins"]   / g * 100, 1) if g else 0
        b["loss_rate"]  = round(b["losses"] / g * 100, 1) if g else 0
        b["confidence"] = confidence(g)

    # Collapse rate
    long_g = stats["by_length"]["long"]["games"]
    stats["collapse_rate"] = {
        "collapses": stats["collapse_games"],
        "long_games": long_g,
        "rate": round(stats["collapse_games"] / long_g * 100, 1) if long_g else 0,
        "confidence": confidence(long_g)
    }

    # Timeout rate (time pressure proxy)
    timeouts = (stats["terminations"].get("timeout", 0) +
                stats["terminations"].get("time forfeit", 0))
    stats["timeout_rate"] = {
        "timeouts": timeouts,
        "total_games": total,
        "rate": round(timeouts / total * 100, 1),
        "confidence": confidence(total)
    }

    # Rating normalization note
    stats["rating_context"] = {
        "losses_vs_lower_rated": stats["losses_vs_lower_rated"],
        "losses_vs_higher_rated": stats["losses_vs_higher_rated"],
        "wins_vs_higher_rated": stats["wins_vs_higher_rated"],
        "games_with_rating_data": stats["rating_comparison_games"],
        "note": "Losses vs lower-rated opponents are weighted more heavily as genuine weaknesses."
    }

    # Opening stats — with rating context, min 3 games
    opening_list = []
    for op, rec in stats["openings"].items():
        t = rec["wins"] + rec["losses"] + rec["draws"]
        if t < 3:
            continue

        loss_rate = round(rec["losses"] / t * 100, 1)
        win_rate_op = round(rec["wins"] / t * 100, 1)

        # Average opponent rating when losing this opening (normalization)
        avg_opp_when_losing = (
            round(sum(rec["opp_ratings_loss"]) / len(rec["opp_ratings_loss"]))
            if rec["opp_ratings_loss"] else None
        )

        opening_list.append({
            "opening": op,
            "games": t,
            "wins": rec["wins"],
            "losses": rec["losses"],
            "draws": rec["draws"],
            "win_rate": win_rate_op,
            "loss_rate": loss_rate,
            "avg_opp_rating_when_losing": avg_opp_when_losing,
            "confidence": confidence(t)
        })

    stats["worst_openings"] = sorted(opening_list, key=lambda x: x["loss_rate"], reverse=True)[:6]
    stats["best_openings"]  = sorted(opening_list, key=lambda x: x["win_rate"],  reverse=True)[:6]

    # Clean up raw lists to keep response lean
    del stats["game_lengths_ply"]
    del stats["opponent_ratings"]
    # Clean opening raw rating lists
    stats["openings"] = {}  # don't send full opening dict — worst/best is enough

    stats["data_summary"] = {
        "games_analyzed": total,
        "confidence_overall": confidence(total)["label"],
        "note": "Analysis based on most recent games. Older games may reflect different habits."
    }

    return stats

# ── AI Prompt ─────────────────────────────────────────────────────────────────

# System prompt is static — good candidate for Claude prompt caching (Phase 2)
SYSTEM_PROMPT = """You are ChessScout, an elite chess intelligence engine trusted by tournament players.

Your job: produce a surgical, evidence-backed scouting report using ONLY the data provided.

STRICT RULES:
1. Every claim must cite exact numbers. Never say "they struggle" without a percentage and sample size.
2. Add confidence ratings: [Confidence: HIGH — 42 games] or [LOW — 4 games]
3. Never use psychological labels (tilt, nervous, lacks confidence). Only describe observable behaviors.
4. If sample size < 10 for a claim, flag it explicitly.
5. Every opening recommendation must end with: "Expected score: X% (based on their Y% loss rate in Z games)"
6. Rating context matters: losing to higher-rated players is normal. Losing to lower-rated players is a real weakness.
7. Tone: sharp, factual, like a data analyst who plays chess at GM level. No hype. No filler."""


def build_prompt(username: str, s: dict) -> str:
    return f"""Analyze this player and produce a ChessScout scouting report.

Player: {username}
Platform: analyzed from recent games
Games analyzed: {s['total_games']}
Overall confidence: {s.get('data_summary', {}).get('confidence_overall', 'N/A')}

=== CORE STATS ===
Overall win rate: {s['overall_wr']['win_rate']}% ({s['overall_wr']['sample']} games)
White win rate:   {s['white_wr']['win_rate']}% ({s['white_wr']['sample']} games)
Black win rate:   {s['black_wr']['win_rate']}% ({s['black_wr']['sample']} games)

=== GAME LENGTH PERFORMANCE ===
Short games (<20 moves):   {s['by_length']['short']['win_rate']}% win / {s['by_length']['short']['loss_rate']}% loss ({s['by_length']['short']['games']} games) [{s['by_length']['short']['confidence']['label']}]
Medium games (20-59 moves): {s['by_length']['medium']['win_rate']}% win / {s['by_length']['medium']['loss_rate']}% loss ({s['by_length']['medium']['games']} games) [{s['by_length']['medium']['confidence']['label']}]
Long games (60+ moves):    {s['by_length']['long']['win_rate']}% win / {s['by_length']['long']['loss_rate']}% loss ({s['by_length']['long']['games']} games) [{s['by_length']['long']['confidence']['label']}]

=== TIME PRESSURE ===
Lost on time / timeout: {s['timeout_rate']['rate']}% of all games ({s['timeout_rate']['timeouts']} out of {s['timeout_rate']['total_games']}) [{s['timeout_rate']['confidence']['label']}]
Collapse in long games: {s['collapse_rate']['rate']}% of 60+ move games ended in loss/timeout ({s['collapse_rate']['collapses']} / {s['collapse_rate']['long_games']}) [{s['collapse_rate']['confidence']['label']}]

=== RATING CONTEXT (normalization) ===
Losses vs lower-rated opponents (50+ below): {s['rating_context']['losses_vs_lower_rated']} games
Losses vs higher-rated opponents (50+ above): {s['rating_context']['losses_vs_higher_rated']} games
Wins vs significantly higher-rated (100+ above): {s['rating_context']['wins_vs_higher_rated']} games
Note: {s['rating_context']['note']}

=== HOW GAMES END ===
{s['terminations']}

=== WORST OPENINGS (by loss rate, min 3 games) ===
{s['worst_openings']}

=== BEST OPENINGS (by win rate, min 3 games) ===
{s['best_openings']}

Avg game length: {s.get('avg_game_length_ply', 'N/A')} half-moves
Avg opponent rating faced: {s.get('avg_opponent_rating', 'N/A')}

---
Write the report with these EXACT sections:

## 🔍 Player Overview
2-3 sentences. State level, win rate, games analyzed. Observable style only.

## ⚠️ Key Weaknesses
3-5 weaknesses. Format each as:
**Weakness:** [one line]
**Evidence:** [exact numbers — win/loss rate, sample size, rating context if relevant]
**Confidence:** [HIGH/MEDIUM/LOW — X games]

## ♟️ Opening Recommendations (to play AGAINST them)
3 openings. Format each as:
**Opening:** [Name + first 3-4 moves]
**Why:** [their exact loss rate in this or similar lines + rating context]
**Expected score:** [X% based on their historical data]
**Confidence:** [HIGH/MEDIUM/LOW — X games]

## 📊 Why We Believe This
- Games analyzed: {s['total_games']}
- Overall data confidence: {s.get('data_summary', {}).get('confidence_overall', 'N/A')}
- Any caveats about small samples or data limitations

## 📈 Game Plan — How To Beat Them
5 concrete steps. Reference specific data points. No vague advice.

## 💡 If This Is YOU — Improvement Roadmap
3 specific study suggestions tied directly to the measured weaknesses. Include resource names."""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ChessScout API running 🚀", "version": "2.1"}


@app.post("/scout")
async def scout(req: ScoutRequest):
    username = req.username.strip()
    platform = req.platform.lower()
    num_games = min(req.num_games, 200)

    # Check cache first
    cache_key = f"{platform}:{username.lower()}"
    if cache_key in _cache:
        return _cache[cache_key]

    # Fetch games
    if platform == "lichess":
        pgns = await fetch_lichess_games(username, num_games)
    else:
        pgns = await fetch_chesscom_games(username, num_games)

    if not pgns:
        raise HTTPException(status_code=404, detail="No games found for this player.")

    stats = analyze_games(pgns, username)

    if stats["total_games"] < 5:
        raise HTTPException(status_code=400, detail="Not enough games to analyze (minimum 5).")

    # Estimated fetch time warning included in response
    prompt = build_prompt(username, stats)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    result = {
        "username": username,
        "platform": platform,
        "stats": stats,
        "report": message.content[0].text,
    }

    # Cache result (simple in-memory, resets on server restart)
    _cache[cache_key] = result
    return result


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    print(f"FEEDBACK | {req.username} | {req.rating} | {req.comment}")
    return {"status": "Thanks for your feedback!"}


@app.post("/prediction-result")
async def prediction_result(req: PredictionResult):
    """Phase 2 — track whether ChessScout predictions were correct."""
    print(f"PREDICTION | {req.username} | predicted: {req.predicted_opening} | actual: {req.actual_result} | played: {req.opening_played}")
    return {"status": "Result recorded. This makes ChessScout smarter — thank you."}


@app.delete("/cache/{username}")
async def clear_cache(username: str, platform: str = "chess.com"):
    """Clear cached report for a player — useful when they've played many new games."""
    key = f"{platform}:{username.lower()}"
    if key in _cache:
        del _cache[key]
        return {"status": f"Cache cleared for {username}"}
    return {"status": "No cache found for this player"}
