"""
Auto-detecting Pavlovian fiber-photometry data processing.

Builds on this package's existing utilities rather than re-reading raw NWB:
    nwb_utils.load_nwb_from_filename / create_df_events / create_df_fip
    alignment.event_triggered_response  (PSTH / ETR engine)

Public functions:
    canonical_event_name
    load_pavlovian_dfs
    detect_paradigm
    classify_trials
    compute_pav_cs_psth
    process_nwb

Visualization lives in pavlovian_plots.py:
    plot_session_overview
    plot_cs_psth_grid
    plot_lick_quant
    plot_reaction_time
    plot_nwb_summary
    analyze_nwb
    ...

The paradigm (how many CS, reward-only vs reward+airpuff) is inferred from the
events table, so the same entry point handles Stage0-3 (and "Custom"):
    Stage1 : 1CS-US (reward)
    Stage2 : 3CS-US (reward)
    Stage3 : 4CS-US (reward / airpuff)
    Stage0 : 2CS-US (reward / airpuff)
"""

import re
import warnings

import numpy as np
import pandas as pd

from . import nwb_utils
from aind_dynamic_foraging_data_utils.alignment import event_triggered_response
from rachel_analysis_utils import data_curation_helpers
from rachel_analysis_utils.nwb_utils import dummy_nwb

# Data-shaping constants live in nwb_utils; re-used here for viz.
canonical_event_name = nwb_utils.canonical_event_name  # re-export for convenience
CHANNEL_MAP = nwb_utils.FIP_CHANNEL_MAP  # {'Iso':'Iso','G':'Green','R':'Red'}
DEFAULT_PREPROCESSING = nwb_utils.DEFAULT_FIP_PREPROCESSING

CHANNEL_ORDER = ["Iso", "Green", "Red"]

# CS -> (US kind, plot color, US-delivered label, omission label)
CS_INFO = {
    "CS1": ("reward", (1.0, 0.0, 0.0), "R+", "R-"),
    "CS2": ("reward", (0.0, 0.7, 0.0), "R+", "R-"),
    "CS3": ("reward", (1.0, 0.0, 1.0), "R+", "R-"),
    "CS4": ("airpuff", (0.3, 0.3, 0.3), "P+", "P-"),
}

# ETR resamples onto this grid by interpolation; it need not equal the FIP
# acquisition rate, it only sets the output resolution (FIP is ~20 Hz).
OUTPUT_SR = 20.0

# anticipatory-lick summary defaults
ANTILICK_WINDOW = (0.0, 2.0)  # seconds after CS onset (CS->US delay)
ANTILICK_SUMMARY_STAT = "mean"  # 'mean' -> mean +/- SD, 'median' -> median +/- IQR
REW_WINDOW = (2.0, 4.0)  # seconds after CS onset for reward/US response


def load_pavlovian_dfs(
    nwb_or_path,
    preprocessing=DEFAULT_PREPROCESSING,
    adjust_time=None,
    channels=None,
    curation=None,
):
    """Load the analysis-ready dataframes for one session.

    Thin orchestration over ``nwb_utils`` (which does all the NWB manipulation:
    ms->s conversion, ``adjust_time`` alignment, canonical event labels, and the
    ``preprocessing`` filter with ``channel``/``roi`` parsing), plus the channel
    filter and fiber curation, so the frames come back ready to plot.

    Parameters
    ----------
    nwb_or_path : str or NWBFile
        Session to read.
    preprocessing : str
        dF/F variant suffix, e.g. ``'dff-bright_mc-iso-IRLS'``.
    adjust_time : bool or None
        Align time to the first CS. ``None`` -> auto (True when the session has a
        ``CS_start_time`` trials column, else False).
    channels : dict or None
        Optional ``{'<Chan>_<ROI>': 'location'}`` filter; ``None`` -> keep all.
    curation : pandas.DataFrame or None
        Fiber curation from ``data_curation_helpers.load_curation``; drops fibers that
        failed curation and labels the survivors by their curated target.

    Returns
    ----------
    df_events : pandas.DataFrame
        Tidy events with ``timestamps`` (s), ``event``, ``trial``, ``canonical``.
    df_fip : pandas.DataFrame
        Tidy FIP for the selected variant with ``channel`` / ``roi``, filtered and
        curated.
    df_trials : pandas.DataFrame
        Per-trial table for this session.
    meta : dict
        ``subject_id``, ``date``, ``adjust_time``, ``ses_idx``, ``nwb_suffix``.
    """
    nwb = nwb_utils.load_nwb_from_filename(nwb_or_path)
    if adjust_time is None:
        adjust_time = nwb_utils.can_align_to_cs(nwb)

    df_events = nwb_utils.create_df_events(nwb, adjust_time=adjust_time, verbose=False)
    df_fip = nwb_utils.create_df_fip(
        nwb, preprocessing=preprocessing, adjust_time=adjust_time, verbose=False
    )
    df_trials = nwb_utils.create_df_trials(nwb, adjust_time=adjust_time, verbose=False)
    subject_id, session_date = nwb_utils.parse_session_name(nwb)
    # ses_idx/nwb_suffix key this recording against a curation CSV. Both are built from
    # session_start_time rather than nwb.session_id, because a derived asset's name carries
    # the processing date and would yield a ses_idx that matches nothing in the CSV.
    start = getattr(nwb, "session_start_time", None)
    meta = {
        "subject_id": subject_id,
        "date": session_date,
        "adjust_time": bool(adjust_time),
        "ses_idx": "%s_%s" % (subject_id, session_date),
        "nwb_suffix": int(start.strftime("%H%M%S")) if start is not None else None,
    }

    df_fip = _filter_channels(df_fip, channels)
    if curation is not None:
        # curation treats a loaded fiber with no CSV row as an error, and create_df_fip
        # loads every channel in the NWB, so narrow to the requested ones first -- Iso in
        # particular appears in no curation CSV. get_nwb_processed filters at load instead.

        # df_events is untouched: curation is keyed per fiber, and licks/CS/reward have no
        # fiber dimension, so they stay valid however many fibers are dropped here.
        df_fip = _apply_curation(df_fip, curation, meta, preprocessing)

    return df_events, df_fip, df_trials, meta


def detect_paradigm(df_events):
    """Infer the paradigm from the canonical events present.

    Returns a dict with ``stage`` (str), ``cs_list`` (list of CS names present),
    and ``has_airpuff`` (bool).
    """
    present = set(df_events["canonical"].dropna().unique())
    active = [c for c in ("CS1", "CS2", "CS3", "CS4") if c in present]
    has_airpuff = "Airpuff" in present

    cs_list = [c for c in active if not (CS_INFO[c][0] == "airpuff" and not has_airpuff)]
    stage = {
        (1, False): "Stage1 (1CS-US, reward)",
        (3, False): "Stage2 (3CS-US, reward)",
        (2, True): "Stage0 (2CS-US, reward/airpuff)",
        (4, True): "Stage3 (4CS-US, reward/airpuff)",
    }.get(
        (len(cs_list), has_airpuff),
        "Custom (%dCS-US%s)" % (len(cs_list), ", airpuff" if has_airpuff else ""),
    )
    return {"stage": stage, "cs_list": cs_list, "has_airpuff": has_airpuff}


def classify_trials(df_events, cs_list):
    """Split each CS's trials into US-delivered (pos) vs omission (neg).

    Uses the events table ``trial`` column: a CS trial is positive when that
    trial number also contains the matching US event (reward or airpuff).

    Returns ``{cs_name: {"onsets": np.ndarray, "pos_mask": np.ndarray(bool),
    "trials": np.ndarray}}`` where ``onsets`` are CS-onset timestamps (s).
    """
    reward_trials = set(df_events.loc[df_events["canonical"] == "Reward", "trial"].astype(int))
    airpuff_trials = set(df_events.loc[df_events["canonical"] == "Airpuff", "trial"].astype(int))
    us_by_kind = {"reward": reward_trials, "airpuff": airpuff_trials}

    out = {}
    for cs in cs_list:
        rows = df_events[df_events["canonical"] == cs].sort_values("timestamps")
        onsets = rows["timestamps"].to_numpy(float)
        trials = rows["trial"].astype(int).to_numpy()
        delivered = us_by_kind[CS_INFO[cs][0]]
        pos_mask = np.array([t in delivered for t in trials], dtype=bool)
        out[cs] = {"onsets": onsets, "pos_mask": pos_mask, "trials": trials}
    return out


def compute_pav_cs_psth(
    df_fip,
    channel,
    roi,
    event_times,
    t_before=5.0,
    t_after=15.0,
    baseline=5.0,
    output_sampling_rate=OUTPUT_SR,
):
    """Event-triggered response for one channel/ROI, baseline-subtracted.

    Wraps ``alignment.event_triggered_response`` and returns
    ``(time, mean, sem, n)`` in percent dF/F. Empty inputs yield zero-length
    ``mean``/``sem`` and ``n == 0``.
    """
    sub = df_fip[(df_fip["channel"] == channel) & (df_fip["roi"] == roi)]
    if len(sub) == 0 or len(event_times) == 0:
        return np.array([]), np.array([]), np.array([]), 0

    data = sub[["timestamps", "data"]].sort_values("timestamps").reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        etr = event_triggered_response(
            data=data,
            t="timestamps",
            y="data",
            event_times=list(event_times),
            t_start=-abs(t_before),
            t_end=abs(t_after),
            output_sampling_rate=output_sampling_rate,
            output_format="tidy",
            interpolate=True,
            censor=False,
            nan_policy="interpolate",
        )
    if etr is None or len(etr) == 0:
        return np.array([]), np.array([]), np.array([]), 0

    # tidy -> matrix [time, event]
    wide = etr.pivot_table(index="time", columns="event_number", values="data")
    t = wide.index.to_numpy(float)
    mat = wide.to_numpy(float) * 100.0  # percent dF/F
    base = t < (t[0] + baseline)
    if base.any():
        mat = mat - np.nanmean(mat[base, :], axis=0, keepdims=True)
    mean = np.nanmean(mat, axis=1)
    n = mat.shape[1]
    sem = np.nanstd(mat, axis=1) / np.sqrt(max(n, 1))
    return t, mean, sem, n


def _channels_present(df_fip, channels=None):
    """Return ordered (channel_label, roi) pairs present, honoring a filter.

    ``channels`` is an optional dict keyed by ``'<Chan>_<ROI>'`` base names.
    """
    if channels:
        wanted = set()
        for key in channels:
            m = re.match(r"(Iso|G|R)_(\d+)$", str(key))
            if m:
                wanted.add((CHANNEL_MAP[m.group(1)], int(m.group(2))))
        pairs = [
            (c, r)
            for (c, r) in sorted(
                set(zip(df_fip["channel"], df_fip["roi"])),
                key=lambda cr: (CHANNEL_ORDER.index(cr[0]), cr[1]),
            )
            if (c, r) in wanted
        ]
    else:
        pairs = sorted(
            set(zip(df_fip["channel"], df_fip["roi"])),
            key=lambda cr: (CHANNEL_ORDER.index(cr[0]), cr[1]),
        )
    return pairs


def _filter_channels(df_fip, channels=None):
    """Restrict df_fip's rows to ``channels``, which elsewhere only narrows derived pairs."""
    if not channels:
        return df_fip
    wanted = set(_channels_present(df_fip, channels))
    keep = [(c, r) in wanted for c, r in zip(df_fip["channel"], df_fip["roi"])]
    return df_fip[keep].copy()


def _apply_curation(df_fip, curation, meta, preprocessing):
    """Drop fibers that failed curation and tag survivors with their intended measurement.

    ``curation`` is the frame from ``data_curation_helpers.load_curation``. It is keyed by
    ``(ses_idx, patch_cord)``, neither of which ``create_df_fip`` produces, so both are
    rebuilt here before handing off to the shared helper.
    """
    suffix = "" if preprocessing == "raw" else "_" + preprocessing
    df = df_fip.copy()
    df["ses_idx"] = meta["ses_idx"]
    df["patch_cord"] = df["event"].astype(str).str.removesuffix(suffix)
    df['preprocessing'] = preprocessing

    df_sess = pd.DataFrame({"ses_idx": [meta["ses_idx"]], "nwb_suffix": [meta["nwb_suffix"]]})
    curation = data_curation_helpers.drop_unchosen_recordings(curation, df_sess)

    out = data_curation_helpers.apply_curation_df_fip(df, curation)
    # apply_curation_df_fip rewrites 'event' to the curated target, falling back to the
    # patch cord for fibers it passes through uncurated (Iso). Only the former is a real
    # label, so leave the latter null and let the plots keep their default naming.
    out["target"] = out["event"].where(out["event"] != out["patch_cord"])
    return out


def _anticipatory_lick_counts(licks, onsets, window_s):
    """Per-trial lick count inside ``[onset+w0, onset+w1)`` for each CS onset."""
    w0, w1 = window_s
    return np.array([int(np.sum((licks >= o + w0) & (licks < o + w1))) for o in onsets], dtype=int)


def _anticipatory_summary(df_events, paradigm, cls, window_s):
    """Per-CS anticipatory-lick numbers for the JSON/return summary."""
    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamps"].to_numpy(float)
    out = {}
    for cs in paradigm["cs_list"]:
        counts = _anticipatory_lick_counts(licks, cls[cs]["onsets"], window_s).astype(float)
        pm = cls[cs]["pos_mask"]
        rec = {
            "anti_window_s": [float(window_s[0]), float(window_s[1])],
            "anti_lick_mean": float(np.mean(counts)) if len(counts) else float("nan"),
            "anti_lick_sd": float(np.std(counts)) if len(counts) else float("nan"),
            "anti_lick_mean_pos": (
                float(np.mean(counts[pm])) if len(counts) and pm.any() else float("nan")
            ),
            "anti_lick_mean_neg": (
                float(np.mean(counts[~pm])) if len(counts) and (~pm).any() else float("nan")
            ),
        }
        out[cs] = rec
    return out


def _build_summary(paradigm, cls, meta, chan_labels, n_roi):
    """Assemble (and print) the numeric summary dict for a session."""
    summary = {
        "subject_id": meta["subject_id"],
        "date": meta["date"],
        "stage": paradigm["stage"],
        "n_roi": n_roi,
        "channels": chan_labels,
        "adjust_time": meta["adjust_time"],
        "cs": {},
    }
    print(
        "[detected] %s | CS=%s | ROIs=%d | channels=%s | adjust_time=%s"
        % (paradigm["stage"], paradigm["cs_list"], n_roi, chan_labels, meta["adjust_time"])
    )
    for cs in paradigm["cs_list"]:
        pm = cls[cs]["pos_mask"]
        summary["cs"][cs] = {
            "n_total": int(len(pm)),
            "n_pos": int(pm.sum()),
            "n_neg": int((~pm).sum()),
        }
        print(
            "  %s: %d trials (%d %s / %d %s)"
            % (cs, len(pm), int(pm.sum()), CS_INFO[cs][2], int((~pm).sum()), CS_INFO[cs][3])
        )
    return summary


def _as_dummy_nwb(df_trials, df_events, df_fip, meta):
    """Bundle one session's frames into a ``rachel_analysis_utils`` ``dummy_nwb``.

    The frames all carry ``ses_idx``, which is what ``dummy_nwb`` keys on, so the
    object is interchangeable with the ones built straight from the foraging pipeline
    (and can be saved/reloaded with the same helpers). ``stage`` is attached on top so
    a session can be labelled without re-reading its events.
    """
    ses_idx = meta["ses_idx"]
    if len(df_fip):
        nwb = dummy_nwb(df_trials, df_events, df_fip, ses_idx=ses_idx)
    else:
        # dummy_nwb asserts the session has FIP rows, but curation may legitimately drop
        # every fiber and the behavioral half stays valid, so build it directly instead
        # (the same way dummy_nwb.load does).
        warnings.warn("no FIP data left for %s; building a behavior-only nwb" % ses_idx)
        nwb = dummy_nwb.__new__(dummy_nwb)
        nwb.session_id = ses_idx
        nwb.df_trials = df_trials
        nwb.df_events = df_events
        nwb.df_fip = df_fip
        nwb.nwb_file_loc = None
    nwb.stage = meta["stage"]
    return nwb


def process_nwb(
    nwb_or_path,
    preprocessing=DEFAULT_PREPROCESSING,
    channels=None,
    curation=None,
    adjust_time=None,
    antilick_window=ANTILICK_WINDOW,
):
    """Load one Pavlovian session into a ``dummy_nwb``, with its numeric summary.

    The data half of :func:`analyze_nwb`: it reads, filters, curates, detects the
    paradigm and classifies trials, but draws nothing. Pass the returned object to
    :func:`pavlovian_plots.plot_nwb_summary` for the single-page figure, save it with
    ``rachel_analysis_utils.nwb_utils`` (``nwb.save`` / ``save_nwb_list``), or collect
    several to plot across sessions. Session-level facts are returned separately in
    ``meta``; collecting those across sessions is what makes a ``df_sess``.

    Parameters
    ----------
    nwb_or_path : str or NWBFile
        A combined behavior+fiber NWB (path or object).
    preprocessing : str
        dF/F variant suffix, e.g. ``'dff-bright_mc-iso-IRLS'``.
    channels : dict or None
        Optional ``{'<Chan>_<ROI>': 'location'}`` filter; None -> all present.
    curation : pandas.DataFrame or None
        Fiber curation from ``data_curation_helpers.load_curation``. Drops fibers that
        failed curation and labels the survivors by their curated target, which takes
        precedence over the intended measurements in ``channels``. A session whose fibers
        are all dropped still yields behavioral frames, with an empty ``df_fip``.
    adjust_time : bool or None
        Align time to the first CS. ``None`` -> auto.
    antilick_window : tuple
        ``(start, end)`` seconds after CS onset for the anticipatory-lick count.

    Returns
    -------
    summary : dict
        Stage, per-CS trial counts (with anticipatory-lick numbers), channels, roi count.
    nwb : rachel_analysis_utils.nwb_utils.dummy_nwb
        Carries ``df_trials``, ``df_events``, ``df_fip`` (already channel-filtered and
        curated) and ``session_id``, plus a ``stage`` attribute.
    meta : dict
        ``load_pavlovian_dfs`` metadata plus ``stage`` and ``cs_list`` from the
        detected paradigm.
    """
    df_events, df_fip, df_trials, meta = load_pavlovian_dfs(
        nwb_or_path, preprocessing, adjust_time, channels=channels, curation=curation
    )

    paradigm = detect_paradigm(df_events)
    cls = classify_trials(df_events, paradigm["cs_list"])
    # carry the paradigm on meta so downstream groupings (dummy_nwb objects, across-session
    # plots) can label a session by stage without re-reading its events
    meta = dict(meta, stage=paradigm["stage"], cs_list=paradigm["cs_list"])

    pairs = _channels_present(df_fip, channels)
    chan_labels = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]
    n_roi = len({r for _, r in pairs})

    summary = _build_summary(paradigm, cls, meta, chan_labels, n_roi)

    # anticipatory-lick numbers into the per-CS summary (also lands in the JSON)
    anti = _anticipatory_summary(df_events, paradigm, cls, antilick_window)
    for cs, rec in anti.items():
        summary["cs"].get(cs, {}).update(rec)

    return summary, _as_dummy_nwb(df_trials, df_events, df_fip, meta), meta


def enrich_df_trials(
    nwb,
    channels=None,
    preprocessing=DEFAULT_PREPROCESSING,
    cs_window=ANTILICK_WINDOW,
    rew_window=REW_WINDOW,
    baseline_window=(-2.0, 0.0),
    antilick_window=ANTILICK_WINDOW,
):
    """Add per-trial FIP response and anticipatory-lick columns to df_trials (in place).

    Accepts either a ``dummy_nwb`` (which already carries pre-computed dataframes as
    ``df_trials``, ``df_events``, ``df_fip`` attributes) or a raw NWBFile — missing
    attributes are built via ``nwb_utils``.

    ``df_trials`` must already have ``CS_type`` and ``CS_start_time_in_session`` columns
    (produced by ``nwb_utils.create_df_trials``). New columns added for CS rows:

      - ``antilick``                  : lick count in ``antilick_window`` after CS onset
      - ``cs_response_<event>``       : mean dF/F in ``cs_window`` after CS onset,
                                        baseline-subtracted (one column per fiber)
      - ``rew_response_<event>``      : mean dF/F in ``rew_window`` after CS onset,
                                        baseline-subtracted (one column per fiber)

    ``<event>`` is the fiber's ``patch_cord`` label (e.g. ``G_0``, ``R_1``).

    All timestamps are in session time (seconds), matching ``df_events["timestamps"]``
    and ``df_fip["timestamps"]``.

    Parameters
    ----------
    nwb : dummy_nwb or NWBFile
        Session object. Attributes ``df_trials``, ``df_events``, ``df_fip`` are used
        when present; otherwise built from the raw NWB via ``nwb_utils``.
    preprocessing : str
        dF/F variant suffix used when ``df_fip`` must be built from a raw NWBFile.
    cs_window : tuple
        ``(t0, t1)`` seconds after CS onset for the CS-response metric.
    rew_window : tuple
        ``(t0, t1)`` seconds after CS onset for the reward-response metric.
    baseline_window : tuple
        ``(t0, t1)`` seconds relative to CS onset used as the pre-event baseline.
    antilick_window : tuple
        ``(t0, t1)`` seconds after CS onset for the anticipatory-lick count.

    Returns
    -------
    pandas.DataFrame
        The enriched df_trials (modified in place when taken from a dummy_nwb).
    """
    df_trials = getattr(nwb, "df_trials", None)
    if df_trials is None:
        df_trials = nwb_utils.create_df_trials(nwb, verbose=False)
    df_events = getattr(nwb, "df_events", None)
    if df_events is None:
        df_events = nwb_utils.create_df_events(nwb, verbose=False)
    df_fip = getattr(nwb, "df_fip", None)
    if df_fip is None:
        df_fip = nwb_utils.create_df_fip(nwb, preprocessing=preprocessing, verbose=False)

    fip_df = _filter_channels(df_fip, channels) if (len(df_fip) > 0 and channels) else df_fip
    events = list(fip_df["event"].unique()) if len(fip_df) > 0 else []

    df_trials["antilick"] = np.nan
    for ev in events:
        df_trials["cs_response_%s" % ev] = np.nan
        df_trials["rew_response_%s" % ev] = np.nan

    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamps"].to_numpy(float)
    b0, b1 = baseline_window
    c0, c1 = cs_window
    r0, r1 = rew_window

    # Pre-slice FIP arrays per event to avoid repeated filtering in the trial loop
    fip_cache = {}
    for ev in events:
        sub = fip_df[fip_df["event"] == ev].sort_values("timestamps")
        fip_cache[ev] = (sub["timestamps"].to_numpy(float), sub["data"].to_numpy(float) * 100.0)

    # Group by CS_type so _anticipatory_lick_counts runs once per CS (vectorized)
    for cs_type, group in df_trials[df_trials["CS_type"].notna()].groupby("CS_type"):
        onsets = group["CS_start_time_in_session"].to_numpy(float)
        idxs = group.index

        df_trials.loc[idxs, "antilick"] = _anticipatory_lick_counts(licks, onsets, antilick_window)

        for ev in events:
            ts, vals = fip_cache[ev]
            cs_resps = np.full(len(onsets), np.nan)
            rew_resps = np.full(len(onsets), np.nan)
            for i, onset in enumerate(onsets):
                base_mask = (ts >= onset + b0) & (ts < onset + b1)
                base = np.nanmean(vals[base_mask]) if base_mask.any() else 0.0
                cs_mask = (ts >= onset + c0) & (ts < onset + c1)
                if cs_mask.any():
                    cs_resps[i] = np.nanmean(vals[cs_mask]) - base
                rew_mask = (ts >= onset + r0) & (ts < onset + r1)
                if rew_mask.any():
                    rew_resps[i] = np.nanmean(vals[rew_mask]) - base
            df_trials.loc[idxs, "cs_response_%s" % ev] = cs_resps
            df_trials.loc[idxs, "rew_response_%s" % ev] = rew_resps

    return df_trials
