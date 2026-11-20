"""gta6-trophy-rarity-forecast-01: an offline statistical forecaster.

Reads make.fun case files (schema makefun/gta6-trophies-case/v1) describing
the Exophase trophy page of Grand Theft Auto VI (PS5): a list of trophies
and a history of page reads, each read giving every trophy's earn rate (the
percent of tracked players who have earned it) plus the tracked-player
count. For every trophy it forecasts five quantiles of the earn rate at the
case's target time and writes them as JSON.

The grader runs ``python3 -P -s /bot/main.py`` once per case, with only that
case's file in the input directory. Outside the grader, MAKEFUN_INPUT_DIR /
MAKEFUN_OUTPUT_DIR stand in for /input and /output. Standard library only,
no network, fully deterministic: no randomness, no wall-clock dependence,
so two runs on the same case are byte-identical.

Model sketch (see forecast_trophy for the full story): Exophase refreshes
its counts only every few hours, so the read history is piecewise constant.
Compress it into *states* (runs of identical reads; a state's timestamp is
the last read that showed it). Work on the logit scale in warped time
x(t) = ln(1 + hours since release midnight), which stretches the frantic
first hours after launch. Per trophy, fit a recency-weighted least-squares
line of logit(rate) on x, shrink its slope towards persistence as the
horizon grows, and anchor the median at the last observed rate plus the
shrunk trend. When enough states exist, blend that median with a
decomposition forecast (fit ln(players) and ln(earners) separately, then
divide). The spread combines a random-walk term, the fit's residual scale
and a floor; a page that has never refreshed gets a wide spread and a small
downward (dilution) nudge to the median.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Quantile levels the contest scores, and the standard normal quantiles at
# exactly those levels (used verbatim, not approximated).
STANDARD_Z = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Earn rates are percents; the logit is taken after clamping into this band
# so a rate of exactly 0 or 100 still has a finite logit.
RATE_LO, RATE_HI = 0.01, 99.99

# Recency weighting: a state 18 h older than the checkpoint counts half.
HALF_LIFE_HOURS = 18.0

# Spread ingredients (all in logit units).
RW_SIGMA_PER_SQRT_DAY = 0.5   # random-walk growth of the logit rate
RESID_FLOOR = 0.12            # the residual scale never goes below this
BASE_SIGMA = 0.10             # irreducible uncertainty, always present
SINGLE_STATE_SIGMA = 0.85     # extra spread when the page never refreshed
FEW_STATES_FACTOR = 1.25      # spread multiplier with fewer than 4 states
SIGMA_LO, SIGMA_HI = 0.25, 2.5

# With a single, never-refreshed state, newly tracked players diluting the
# rate are the dominant risk: nudge the median down this much per day of
# horizon (pro-rated, capped at one day).
DILUTION_PER_DAY = 0.35

# Forecast for a trophy never observed at all: low and wide, standard levels.
NEVER_SEEN_Q = [0.01, 0.02, 0.05, 0.15, 0.5]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def clamp_rate(pct):
    """Clamp an earn-rate percent into [0.01, 99.99] for the logit."""
    return min(RATE_HI, max(RATE_LO, pct))


def logit(pct):
    """Log-odds of a percent, after clamping (finite even for 0 or 100)."""
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid_pct(v):
    """Inverse logit, back to percent, computed without overflow."""
    if v >= 0.0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int.

    Mirrors the kit starter's rounding, so 100 serialises as ``100``
    rather than ``100.0``.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def is_number(v):
    """True for a finite JSON number (bools are not numbers here)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def _poly(coeffs, x):
    """Horner evaluation of a polynomial with coefficients coeffs."""
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc


# Acklam's rational approximation of the inverse normal CDF, used only for
# quantile levels outside the standard five (which use exact z values).
_ACKLAM_A = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
             138.3577518672690, -30.66479806614716, 2.506628277459239]
_ACKLAM_B = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
             66.80131188771972, -13.28068155288572]
_ACKLAM_C = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
             -2.549732539343734, 4.374664141464968, 2.938163982698783]
_ACKLAM_D = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
             3.754408661907416]


def _normsinv_acklam(p):
    """Acklam's rational approximation of the normal quantile at p."""
    if p < 0.02425:  # lower tail
        q = math.sqrt(-2.0 * math.log(p))
        return _poly(_ACKLAM_C, q) / (_poly(_ACKLAM_D, q) * q + 1.0)
    if p > 0.97575:  # upper tail, by symmetry
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -_poly(_ACKLAM_C, q) / (_poly(_ACKLAM_D, q) * q + 1.0)
    q = p - 0.5  # central region
    return _poly(_ACKLAM_A, q * q) * q / (_poly(_ACKLAM_B, q * q) * (q * q) + 1.0)


def normsinv(p):
    """Standard normal quantile at p in (0, 1), Acklam + one Halley step.

    Acklam's approximation alone is accurate to about 1e-9; a single
    Halley refinement against the exact CDF (via math.erf) brings it to
    near machine precision, which keeps non-standard quantile levels as
    trustworthy as the exact standard five. The step is skipped if it
    would overflow in the far tails, where Acklam is already ample.
    """
    p = min(1.0 - 1e-12, max(1e-12, p))
    x = _normsinv_acklam(p)
    try:
        err = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) - p
        u = err * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
        refined = x - u / (1.0 + x * u / 2.0)
        if math.isfinite(refined):
            return refined
    except OverflowError:
        pass
    return x


def z_for_level(level):
    """Exact z for the standard levels, Acklam approximation otherwise."""
    return STANDARD_Z[level] if level in STANDARD_Z else normsinv(level)


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------

def parse_ts(value):
    """Parse an ISO timestamp (trailing 'Z' ok) to an aware UTC datetime.

    Returns None for anything unparseable. Naive values are read as UTC,
    and a bare date ('2026-11-19') parses to that midnight.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def x_of(t, t0):
    """Warped time: ln(1 + hours from the release anchor t0 to t).

    Reads a little before release give a negative hour count; the log's
    argument is floored at 0.05 so very early reads stay finite and cannot
    dominate a fit as extreme outliers.
    """
    hours = (t - t0).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def recency_weight(t_state, t_checkpoint):
    """0.5 ** (age in hours / 18): exponential decay, 18 h half-life."""
    age_hours = (t_checkpoint - t_state).total_seconds() / 3600.0
    return 0.5 ** (age_hours / HALF_LIFE_HOURS)


# ---------------------------------------------------------------------------
# Weighted least squares on (x, y) points
# ---------------------------------------------------------------------------

def wls_fit(xs, ys, ws):
    """Weighted least squares of y on x -> (x_bar, y_bar, slope).

    The slope is 0 when the x's have no weighted spread, in which case
    predictions reduce to the weighted mean y_bar.
    """
    sw = sum(ws)
    x_bar = sum(w * x for x, w in zip(xs, ws)) / sw
    y_bar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return x_bar, y_bar, 0.0
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in zip(xs, ys, ws))
    return x_bar, y_bar, sxy / sxx


def wls_predict(fit, x):
    """Predict y at x from a wls_fit result."""
    x_bar, y_bar, slope = fit
    return y_bar + slope * (x - x_bar)


def wls_resid_rms(xs, ys, ws, fit):
    """Weighted root-mean-square residual of a wls_fit."""
    total = sum(w * (y - wls_predict(fit, x)) ** 2
                for x, y, w in zip(xs, ys, ws))
    return math.sqrt(total / sum(ws))


# ---------------------------------------------------------------------------
# History compression: reads -> states
# ---------------------------------------------------------------------------

def compress_states(history, fallback_t):
    """Collapse runs of identical reads into states.

    A new state starts when (players_tracked, rates) differs from the
    previous read. A state's timestamp is the LAST read time showing that
    state; its players/rates are that state's (shared) values.
    """
    states, prev_key = [], None
    for read in history:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        rates = list(rates) if isinstance(rates, list) else []
        players = read.get("players_tracked")
        t = parse_ts(read.get("at") or read.get("slot")) or fallback_t
        key = (players if is_number(players) else None,
               tuple(v if is_number(v) else None for v in rates))
        if states and key == prev_key:
            states[-1]["t"] = t  # same state, seen again later
        else:
            states.append({"t": t, "players": players, "rates": rates})
            prev_key = key
    return states


# ---------------------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------------------

def forecast_case(case):
    """Forecast one parsed case dict -> {"trophies": [{id, q}, ...]}."""
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [z_for_level(float(lv)) for lv in levels]

    # Anchor: release day at 00:00:00Z. If the release date is unusable,
    # fall back to midnight of the checkpoint day so x stays meaningful.
    checkpoint = parse_ts(case.get("at")) or parse_ts(case.get("target"))
    game = case.get("game") or {}
    t0 = parse_ts(game.get("release_date")) if isinstance(game, dict) else None
    if t0 is None:
        base = checkpoint or parse_ts(case.get("target"))
        t0 = base.replace(hour=0, minute=0, second=0, microsecond=0) \
            if base is not None else datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Horizon: prefer the stated minutes; derive it from target - checkpoint
    # when missing so the spread still scales with the wait.
    target = parse_ts(case.get("target"))
    if is_number(case.get("horizon_minutes")):
        horizon_minutes = float(case["horizon_minutes"])
    elif target is not None and checkpoint is not None:
        horizon_minutes = (target - checkpoint).total_seconds() / 60.0
    else:
        horizon_minutes = 0.0
    horizon_minutes = max(horizon_minutes, 0.0)
    if checkpoint is None:
        checkpoint = target or t0
    if target is None:
        target = checkpoint
    x_target = x_of(target, t0)

    states = compress_states(history, checkpoint)
    state_xs = [x_of(s["t"], t0) for s in states]
    state_ws = [recency_weight(s["t"], checkpoint) for s in states]

    # Players tracked in the very last read (== last state's), for caps.
    last_players = None
    if history and isinstance(history[-1], dict) \
            and is_number(history[-1].get("players_tracked")):
        last_players = float(history[-1]["players_tracked"])
    elif states and is_number(states[-1]["players"]):
        last_players = float(states[-1]["players"])

    sigma_rw = RW_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_minutes / 1440.0)
    rows = []
    for idx, trophy in enumerate(trophies):
        rows.append({"id": trophy["id"], "q": forecast_trophy(
            idx, history, states, state_xs, state_ws, last_players,
            x_target, horizon_minutes, sigma_rw, zs)})
    return {"trophies": rows}


def forecast_trophy(idx, history, states, state_xs, state_ws, last_players,
                    x_target, horizon_minutes, sigma_rw, zs):
    """Five quantiles for the trophy at position idx (or NEVER_SEEN_Q)."""
    n_states = len(states)
    # r_last: the rate in the case's last history read; if that read lacks
    # the trophy (null), the last non-null state rate; if none ever, the
    # fixed never-seen forecast.
    r_last = None
    if history and isinstance(history[-1], dict):
        last_rates = history[-1].get("rates") or []
        if idx < len(last_rates) and is_number(last_rates[idx]):
            r_last = float(last_rates[idx])
    # Series over states where this trophy was actually read.
    series = [(sx, float(s["rates"][idx]), w)
              for s, sx, w in zip(states, state_xs, state_ws)
              if idx < len(s["rates"]) and is_number(s["rates"][idx])]
    if r_last is None and series:
        r_last = series[-1][1]
    if r_last is None:
        return list(NEVER_SEEN_Q)

    m = logit(r_last)
    resid = 0.0

    # Trend fit: WLS of logit(rate) on x, with the slope shrunk towards
    # persistence by span / (span + horizon) on the warped time axis. The
    # median stays anchored at the last rate; only the trend moves it.
    if len(series) >= 2:
        xs = [p[0] for p in series]
        if len(set(xs)) >= 2 and len(set(p[1] for p in series)) >= 2:
            ys = [logit(p[1]) for p in series]
            ws = [p[2] for p in series]
            fit = wls_fit(xs, ys, ws)
            resid = wls_resid_rms(xs, ys, ws, fit)
            span_x = xs[-1] - xs[0]
            horizon_x = max(0.0, x_target - xs[-1])
            shrink = span_x / (span_x + horizon_x) \
                if span_x + horizon_x > 0 else 0.0
            m = logit(r_last) + fit[2] * shrink * horizon_x

    # Decomposition ensemble (needs >= 3 states, trophy earned in >= 3):
    # forecast earners and players separately, then divide. Blended 50/50
    # with the trend median on the logit scale.
    positive = sum(1 for s in states
                   if idx < len(s["rates"]) and is_number(s["rates"][idx])
                   and s["rates"][idx] > 0)
    if n_states >= 3 and positive >= 3:
        blended = decomposition_median(idx, states, state_xs, state_ws,
                                       x_target, last_players)
        if blended is not None:
            m = 0.5 * m + 0.5 * blended

    # Spread: random walk + fit residuals + a floor. A page that has never
    # refreshed can move all at once at the next refresh, and fresh players
    # mostly dilute, hence the wide spread and the downward median nudge.
    if n_states == 1:
        sigma = math.sqrt(sigma_rw ** 2 + SINGLE_STATE_SIGMA ** 2)
        m -= DILUTION_PER_DAY * min(1.0, horizon_minutes / 1440.0)
    else:
        sigma = math.sqrt(sigma_rw ** 2 + max(resid, RESID_FLOOR) ** 2
                          + BASE_SIGMA ** 2)
    if n_states < 4:
        sigma *= FEW_STATES_FACTOR
    sigma = min(SIGMA_HI, max(SIGMA_LO, sigma))

    # Quantiles: normal on the logit scale, rounded half-up to 4 decimals,
    # then forced non-decreasing by cumulative maximum.
    q = [round4(sigmoid_pct(m + z * sigma)) for z in zs]
    for j in range(1, len(q)):
        if q[j] < q[j - 1]:
            q[j] = q[j - 1]
    return q


def decomposition_median(idx, states, state_xs, state_ws, x_target,
                         last_players):
    """Logit of the earners/players decomposition forecast, or None.

    Fits ln(players_tracked) on x over all states and ln(earners) on x over
    the states where this trophy's earners are positive, predicts both at
    the target and divides. The player prediction is floored at 80% of the
    last tracked count (and at 1): the tracked population cannot plausibly
    collapse between the checkpoint and the target.
    """
    try:
        p_pts = [(sx, math.log(float(s["players"])), w)
                 for s, sx, w in zip(states, state_xs, state_ws)
                 if is_number(s.get("players")) and s["players"] > 0]
        if len(p_pts) < 2:
            return None
        p_fit = wls_fit([p[0] for p in p_pts], [p[1] for p in p_pts],
                        [p[2] for p in p_pts])
        p_hat = math.exp(max(-50.0, min(50.0, wls_predict(p_fit, x_target))))
        if last_players is not None:
            p_hat = max(p_hat, last_players * 0.8)
        p_hat = max(p_hat, 1.0)

        e_pts = []
        for s, sx, w in zip(states, state_xs, state_ws):
            if idx < len(s["rates"]) and is_number(s["rates"][idx]) \
                    and s["rates"][idx] > 0 and is_number(s.get("players")) \
                    and s["players"] > 0:
                earners = float(s["rates"][idx]) / 100.0 * float(s["players"])
                if earners > 0:
                    e_pts.append((sx, math.log(earners), w))
        if len(e_pts) < 2:
            return None
        e_fit = wls_fit([p[0] for p in e_pts], [p[1] for p in e_pts],
                        [p[2] for p in e_pts])
        e_hat = max(math.exp(max(-50.0, min(50.0,
                                            wls_predict(e_fit, x_target)))),
                    0.0)
        return logit(min(RATE_HI, max(RATE_LO, 100.0 * e_hat / p_hat)))
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Fallback for malformed cases: plain persistence, starter-style
# ---------------------------------------------------------------------------

def fallback_case(case):
    """Persistence forecast from whatever of the case is still usable.

    Used when forecast_case raises, so a malformed case still yields an
    output file instead of none. Trophies without a usable last rate get
    the never-seen quantiles.
    """
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [z_for_level(float(lv)) for lv in levels]
    horizon = case.get("horizon_minutes")
    spread = RW_SIGMA_PER_SQRT_DAY * math.sqrt(
        (float(horizon) if is_number(horizon) else 0.0) / 1440.0)
    last_rates = []
    if history and isinstance(history[-1], dict):
        last_rates = history[-1].get("rates") or []
    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy.get("id") if isinstance(trophy, dict) else idx + 1
        rate = last_rates[idx] if idx < len(last_rates) else None
        if is_number(rate):
            centre = logit(float(rate))
            q = [round4(sigmoid_pct(centre + z * spread)) for z in zs]
            for j in range(1, len(q)):
                if q[j] < q[j - 1]:
                    q[j] = q[j - 1]
        else:
            q = list(NEVER_SEEN_Q)
        rows.append({"id": tid, "q": q})
    return {"trophies": rows}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.is_dir():
        return
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(input_dir.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            case = {}
        try:
            forecast = forecast_case(case)
        except Exception:
            try:
                forecast = fallback_case(case)
            except Exception:
                forecast = {"trophies": []}
        (output_dir / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
