"""
Pavlovian fiber-photometry visualization.

Plotting counterpart to pavlovian_analysis.py. All functions here consume
the dataframes and dummy_nwb objects produced by process_nwb.

Public functions:
    plot_session_overview
    plot_pavlovian_session_plotly
    plot_pavlovian_session_nwb_plotly
    plot_cs_psth_grid
    plot_cs_psth_compare
    plot_lick_quant
    plot_anticipatory_lick_summary
    plot_reaction_time
    plot_cs_response_summary
    psth_CS_fip
    render_summary_figure
    plot_nwb_summary
"""

import warnings

import matplotlib

matplotlib.use("Agg")  # capsule/headless safe; callers may override before import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import pandas as pd  # noqa: E402

from .nwb_utils import parse_session_name  # noqa: E402
from .pavlovian_analysis import (  # noqa: E402
    CHANNEL_MAP,
    CHANNEL_ORDER,
    CS_INFO,
    OUTPUT_SR,
    ANTILICK_WINDOW,
    ANTILICK_SUMMARY,
    REW_WINDOW,
    DEFAULT_PREPROCESSING,
    _channels_present,
    _anticipatory_lick_counts,
    compute_pav_cs_psth,
    detect_paradigm,
    classify_trials,
)

CHANNEL_COLOR = {"Iso": "blue", "Green": "green", "Red": "magenta"}

# default PSTH window (seconds, relative to CS onset)
T_BEFORE = 5.0
T_AFTER = 15.0
BASELINE = 5.0
# compare-across-CS panel window (seconds, relative to CS onset)
CMP_T_BEFORE = 1.0
CMP_T_AFTER = 5.0
CMP_BASELINE = 1.0
PNG_DPI = 150  # resolution of the PNG export

ANTILICK_SWARM_WIDTH = 0.28


def _pair_labels(df_fip):
    """Map each (channel, roi) to its curated target, empty when no curation was applied."""
    if "target" not in df_fip or not len(df_fip):
        return {}
    cols = df_fip[["channel", "roi", "target"]].dropna().drop_duplicates()
    return {(c, r): str(t) for c, r, t in cols.itertuples(index=False)}


def _roi_label(labels, roi, chans):
    """Row label for one ROI: its curated targets when present, else plain 'ROIn'."""
    targets = [labels[(c, roi)] for c in chans if (c, roi) in labels]
    if not targets:
        return "ROI%d" % roi
    return "ROI%d - %s" % (roi, ", ".join(dict.fromkeys(targets)))


def plot_session_overview(df_events, df_fip, paradigm, meta, channels=None, fig=None):
    """Whole-session traces for every channel/ROI with CS/US/lick markers.

    Draws into ``fig`` (a Figure or SubFigure) when given, else creates one.
    """
    pairs = _channels_present(df_fip, channels)
    rois = sorted({r for _, r in pairs})
    chans = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]
    labels = _pair_labels(df_fip)

    if fig is None:
        fig = plt.figure(figsize=(20, 3 + 1.2 * max(len(rois), 1)))
    ax = fig.subplots()
    for i, roi in enumerate(rois):
        off = -i * 100
        for c in chans:
            sub = df_fip[(df_fip["channel"] == c) & (df_fip["roi"] == roi)]
            sub = sub.sort_values("timestamps")
            if len(sub):
                ax.plot(sub["timestamps"], sub["data"] * 100 + off, color=CHANNEL_COLOR[c], lw=0.6)
        ax.axhline(off, ls="--", color="k", lw=0.5)
        ax.text(
            df_fip["timestamps"].max(),
            off,
            "  " + _roi_label(labels, roi, chans),
            va="center",
            fontsize=8,
        )

    def _mark(key, color, width, label):
        """Shade spans for a canonical event key and add one legend proxy."""
        times = df_events.loc[df_events["canonical"] == key, "timestamps"].to_numpy(float)
        for tt in times:
            ax.axvspan(tt, tt + width, color=color)
        if len(times):
            ax.axvspan(0, 0, color=color, label=label)

    _mark("Reward", (0, 0, 1, 0.35), 0.5, "Reward")
    _mark("Airpuff", (0, 0, 0, 0.35), 0.5, "Airpuff")
    for cs in paradigm["cs_list"]:
        _mark(cs, CS_INFO[cs][1] + (0.30,), 1.0, cs)

    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamps"].to_numpy(float)
    if len(licks):
        ax.plot(
            licks,
            np.full(len(licks), 100),
            marker=3,
            ms=6,
            ls="none",
            color=(0, 0, 0, 0.5),
            label="Lick",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F (%)  (ROIs offset)")
    ax.set_title(
        "Session overview — %s %s [%s]" % (meta["subject_id"], meta["date"], paradigm["stage"]),
        fontsize=10,
    )
    ax.grid(True)
    ax.legend(loc="upper right", ncol=6, fontsize=8)
    return fig


def _rgba(rgb_tuple, alpha):
    r, g, b = rgb_tuple[:3]
    return "rgba(%d,%d,%d,%.2f)" % (int(r * 255), int(g * 255), int(b * 255), alpha)


def _add_spans(df_events, shapes, fig, key, rgb, alpha, span_width):
    times = df_events.loc[df_events["canonical"] == key, "timestamps"].to_numpy(float)
    color = _rgba(rgb, alpha)
    for tt in times:
        shapes.append(
            dict(
                type="rect", xref="x", yref="paper",
                x0=tt, x1=tt + span_width, y0=0, y1=1,
                fillcolor=color, line_width=0, layer="below",
            )
        )
    if len(times):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=key, showlegend=True,
            )
        )


def plot_pavlovian_session_plotly(df_events, df_fip, paradigm, meta, channels=None):
    """Plotly session overview: FIP channels stacked above a behavior panel.

    Parameters
    ----------
    df_events : pandas.DataFrame
        Tidy events with ``timestamps`` (s) and ``canonical`` columns.
    df_fip : pandas.DataFrame
        Tidy FIP measurements with ``channel``, ``roi``, ``timestamps``, ``data`` columns.
    paradigm : dict
        Output of :func:`detect_paradigm` (``stage``, ``cs_list``, ``has_airpuff``).
    meta : dict
        Session metadata with ``subject_id`` and ``date`` keys.
    channels : dict or None
        Optional ``{'<Chan>_<ROI>': 'location'}`` filter; ``None`` shows all present.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    BEHAVIOR_BOTTOM = 0.0
    BEHAVIOR_TOP = 1.0
    LICK_CENTER = 0.5
    TICK_HALF = 0.08
    FIP_START = BEHAVIOR_TOP + 0.1
    FIP_GAP = 0.3
    CHANNEL_HEIGHT = 2.0  # fixed display units per channel

    fig = go.Figure()

    # --- Behavior: CS/US event spans ---
    shapes = []
    _add_spans(df_events, shapes, fig, "Reward", (0, 0, 1), 0.85, 0.5)
    _add_spans(df_events, shapes, fig, "Airpuff", (0, 0, 0), 0.70, 0.5)
    for cs in paradigm["cs_list"]:
        _add_spans(df_events, shapes, fig, cs, CS_INFO[cs][1], 0.80, 1.0)
    fig.update_layout(shapes=shapes)

    # Lick ticks
    lick_times = df_events.loc[df_events["canonical"] == "Lick", "timestamps"].to_numpy(float)
    if len(lick_times):
        xs, ys = [], []
        for t in lick_times:
            xs += [t, t, None]
            ys += [LICK_CENTER - TICK_HALF, LICK_CENTER + TICK_HALF, None]
        fig.add_trace(
            go.Scattergl(
                x=xs, y=ys, mode="lines",
                line=dict(color="black", width=1.5),
                name="Lick",
            )
        )

    yticks = [LICK_CENTER]
    ylabels = ["Lick"]

    # --- FIP channels stacked above behavior ---
    # Each channel is mapped to a fixed CHANNEL_HEIGHT band.  Y-tick labels show
    # the actual p01 / p99 data values so the dF/F scale is readable.
    pairs = _channels_present(df_fip, channels)
    band = 0.0
    y_main_top = BEHAVIOR_TOP
    fip_unclipped_ids = []
    fip_clipped_ids = []

    for channel, roi in pairs:
        sub = df_fip[(df_fip["channel"] == channel) & (df_fip["roi"] == roi)].sort_values("timestamps")
        vals = sub["data"].astype(float).to_numpy()
        if vals.size == 0 or np.all(np.isnan(vals)):
            continue

        p01, p99 = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
        span = p99 - p01
        if np.isnan(span) or span == 0:
            span = 1.0

        base = FIP_START + band
        scale = CHANNEL_HEIGHT / span
        d = (vals - p01) * scale + base
        d_clipped = (np.clip(vals, p01, p99) - p01) * scale + base

        color = CHANNEL_COLOR[channel]
        label = f"{channel} ROI{roi}"
        ts = sub["timestamps"].to_numpy()
        custom = np.stack([ts, vals], axis=-1)
        hover = f"%{{customdata[0]:.2f}}s  %{{customdata[1]:.4f}}<extra>{label}</extra>"

        fip_unclipped_ids.append(len(fig.data))
        fig.add_trace(
            go.Scattergl(
                x=ts, y=d, customdata=custom,
                mode="lines", hovertemplate=hover,
                line=dict(color=color, width=1),
                name=label,
            )
        )

        fip_clipped_ids.append(len(fig.data))
        fig.add_trace(
            go.Scattergl(
                x=ts, y=d_clipped, customdata=custom,
                mode="lines", hovertemplate=hover,
                line=dict(color=color, width=1),
                name=label, visible=False, showlegend=False,
            )
        )

        yticks.extend([base, base + CHANNEL_HEIGHT / 2.0, base + CHANNEL_HEIGHT])
        ylabels.extend([f"{p01:.3f}", f"{label}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;", f"{p99:.3f}"])

        band += CHANNEL_HEIGHT + FIP_GAP
        y_main_top = base + CHANNEL_HEIGHT + 0.25

    # Toggle: full range ↔ 1–99% clipped
    if fip_unclipped_ids:
        n_all = len(fig.data)
        vis_full = [True] * n_all
        vis_clip = [True] * n_all
        for idx in fip_clipped_ids:
            vis_full[idx] = False
        for idx in fip_unclipped_ids:
            vis_clip[idx] = False
        for idx in fip_clipped_ids:
            vis_clip[idx] = True
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", direction="right",
                x=1.0, y=1.0, xanchor="right", yanchor="top",
                showactive=True,
                buttons=[
                    dict(label="FIP: full range", method="restyle", args=[{"visible": vis_full}]),
                    dict(label="FIP: clip 1–99%", method="restyle", args=[{"visible": vis_clip}]),
                ],
            )]
        )

    fig.update_yaxes(
        tickvals=yticks, ticktext=ylabels,
        fixedrange=True,
        range=[BEHAVIOR_BOTTOM - 0.05, y_main_top],
    )
    fig.update_xaxes(
        title_text="Time (s)",
        rangeslider=dict(visible=True, thickness=0.06),
    )
    fig.update_layout(
        title=dict(
            text="Session overview — %s %s [%s]"
            % (meta.get("subject_id", ""), meta.get("date", ""), paradigm["stage"]),
            font=dict(size=12),
            x=0.0, xanchor="left", y=0.98, yanchor="top",
        ),
        height=max(400, 300 + 120 * max(len(pairs), 1)),
        width=1000,
        template="simple_white",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=9), entrywidthmode="pixels", entrywidth=125,
        ),
        margin=dict(l=70, r=20, t=120, b=40),
    )
    return fig


def plot_pavlovian_session_nwb_plotly(nwb_list, channels=None):
    """NWB-level entry point for :func:`plot_pavlovian_session_plotly`.

    Complements ``plot_session_in_time_nwb_plotly`` (foraging sessions) for
    Pavlovian fiber-photometry data.  Each item in ``nwb_list`` must expose
    ``df_events`` and ``df_fip`` as attributes; the first item with a ``meta``
    attribute is used as the session metadata in the figure title.

    When more than one NWB is supplied the dataframes are concatenated so the
    full set of events and FIP traces is shown in a single figure.

    Parameters
    ----------
    nwb_list : object or list of objects
        NWB-like objects with ``df_events`` (``timestamps`` + ``canonical``),
        ``df_fip`` (``channel`` / ``roi`` / ``timestamps`` / ``data``), and
        optionally ``meta`` (``subject_id``, ``date``) attributes.
    channels : dict or None
        Optional ``{'<Chan>_<ROI>': 'location'}`` filter; ``None`` -> all present.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not isinstance(nwb_list, (list, tuple)):
        nwb_list = [nwb_list]

    events_acc, fip_acc = [], []
    primary_meta = None

    for nwb in nwb_list:
        if hasattr(nwb, "df_events") and nwb.df_events is not None:
            events_acc.append(nwb.df_events.copy())
        if hasattr(nwb, "df_fip") and nwb.df_fip is not None:
            fip_acc.append(nwb.df_fip.copy())
        if primary_meta is None and hasattr(nwb, "meta") and nwb.meta is not None:
            primary_meta = nwb.meta

    if not events_acc:
        raise ValueError("No df_events found on any nwb in nwb_list")
    if not fip_acc:
        raise ValueError("No df_fip found on any nwb in nwb_list")

    df_events = pd.concat(events_acc, ignore_index=True)
    df_fip = pd.concat(fip_acc, ignore_index=True)
    paradigm = detect_paradigm(df_events)
    return plot_pavlovian_session_plotly(df_events, df_fip, paradigm, primary_meta or {}, channels)


def plot_cs_psth_grid(
    df_fip,
    cs,
    cls,
    meta,
    channels=None,
    t_before=T_BEFORE,
    t_after=T_AFTER,
    baseline=BASELINE,
    output_sampling_rate=OUTPUT_SR,
    fig=None,
):
    """PSTH grid for one CS: rows = ROI, cols = channel, pos vs neg overlaid.

    Draws into ``fig`` (a Figure or SubFigure) when given, else creates one.
    """
    pairs = _channels_present(df_fip, channels)
    rois = sorted({r for _, r in pairs})
    chans = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]
    labels = _pair_labels(df_fip)
    us_kind, cs_color, pos_lab, neg_lab = CS_INFO[cs]

    onsets = cls[cs]["onsets"]
    pos_mask = cls[cs]["pos_mask"]
    pos_t, neg_t = onsets[pos_mask], onsets[~pos_mask]

    if fig is None:
        fig = plt.figure(figsize=(4.5 * len(chans), 3 * max(len(rois), 1)))
    axes = fig.subplots(len(rois), len(chans), squeeze=False)
    fig.suptitle(
        "%s %s  —  %s PSTH (%s vs %s)  n+=%d n-=%d"
        % (
            meta["subject_id"],
            meta["date"],
            cs,
            pos_lab,
            neg_lab,
            int(pos_mask.sum()),
            int((~pos_mask).sum()),
        ),
        fontsize=13,
    )

    for ci, c in enumerate(chans):
        for ri, roi in enumerate(rois):
            ax = axes[ri][ci]
            for times, main, sub, lab in (
                (pos_t, cs_color, cs_color, pos_lab),
                (neg_t, "k", "gray", neg_lab),
            ):
                t, mean, sem, n = compute_pav_cs_psth(
                    df_fip, c, roi, times, t_before, t_after, baseline, output_sampling_rate
                )
                if n > 0:
                    ax.plot(t, mean, color=main, label="%s (n=%d)" % (lab, n))
                    ax.fill_between(t, mean - sem, mean + sem, color=sub, alpha=0.35)
            ax.axvspan(0, 1, color=cs_color + (0.20,))
            ax.axhline(0, color="gray", ls="--", lw=0.6)
            ax.set_xlim(-t_before, t_after)
            ax.grid(True)
            if labels.get((c, roi)):
                ax.set_title(labels[(c, roi)], fontsize=9)
            elif ri == 0:
                ax.set_title(c)
            if ci == 0:
                ax.set_ylabel("ROI%d\ndF/F (%%)" % roi)
            if ri == len(rois) - 1:
                ax.set_xlabel("Time - CS (s)")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
    return fig


def plot_cs_psth_compare(
    df_fip,
    paradigm,
    cls,
    meta,
    channels=None,
    t_before=CMP_T_BEFORE,
    t_after=CMP_T_AFTER,
    baseline=CMP_BASELINE,
    output_sampling_rate=OUTPUT_SR,
    fig=None,
):
    """All CS side-by-side in one row per channel/ROI, shared y-axis per row.

    Same data/colors as :func:`plot_cs_psth_grid` (US-delivered solid, omission
    dashed), but columns are CS so they can be compared directly. Draws into
    ``fig`` (a Figure or SubFigure) when given, else creates one.
    """
    cs_list = paradigm["cs_list"]
    pairs = _channels_present(df_fip, channels)
    if not cs_list or not pairs:
        return None
    labels = _pair_labels(df_fip)
    chans = [c for c in CHANNEL_ORDER if any(cc == c for cc, _ in pairs)]

    if fig is None:
        fig = plt.figure(figsize=(3.0 * len(cs_list), 3.0 * len(pairs)))
    axes = fig.subplots(len(pairs), len(cs_list), squeeze=False, sharey="row")
    fig.suptitle(
        "%s %s  —  CS comparison (%.0f to %.0f s)"
        % (meta["subject_id"], meta["date"], -abs(t_before), abs(t_after)),
        fontsize=13,
    )

    for ri, (chan, roi) in enumerate(pairs):
        for ci, cs in enumerate(cs_list):
            ax = axes[ri][ci]
            _, cs_color, pos_lab, neg_lab = CS_INFO[cs]
            onsets = cls[cs]["onsets"]
            pos_mask = cls[cs]["pos_mask"]
            for times, ls, lab in (
                (onsets[pos_mask], "-", pos_lab),
                (onsets[~pos_mask], "--", neg_lab),
            ):
                t, mean, sem, n = compute_pav_cs_psth(
                    df_fip, chan, roi, times, t_before, t_after, baseline, output_sampling_rate
                )
                if n > 0:
                    ax.plot(t, mean, color=cs_color, ls=ls, label="%s (n=%d)" % (lab, n))
                    ax.fill_between(t, mean - sem, mean + sem, color=cs_color, alpha=0.25, lw=0)
            ax.axvspan(0, 1, color=cs_color + (0.15,))
            ax.axhline(0, color="gray", ls="--", lw=0.6)
            ax.set_xlim(-abs(t_before), abs(t_after))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 0:
                ax.set_title(cs, fontsize=9)
            ax.set_xlabel("Time - CS (s)")
            if ci == 0:
                ax.set_ylabel("%s %s\ndF/F (%%)" % (_roi_label(labels, roi, chans), chan))
            ax.legend(fontsize=7, frameon=False)
    return fig


def plot_lick_quant(df_events, paradigm, cls, fig=None):
    """Anticipatory vs consummatory lick counts per trial, per CS.

    Draws into ``fig`` (a Figure or SubFigure) when given, else creates one.
    """
    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamps"].to_numpy(float)
    if len(licks) == 0 or len(paradigm["cs_list"]) == 0:
        return None
    cs_list = paradigm["cs_list"]
    if fig is None:
        fig = plt.figure(figsize=(5 * len(cs_list), 4))
    axes = fig.subplots(1, len(cs_list), squeeze=False)
    for j, cs in enumerate(cs_list):
        ax = axes[0][j]
        onsets = cls[cs]["onsets"]
        pos_mask = cls[cs]["pos_mask"]
        anti = np.array([np.sum((licks > o) & (licks < o + 2.0)) for o in onsets])
        post = np.array([np.sum((licks > o + 2.0) & (licks < o + 7.0)) for o in onsets])
        ax.plot(anti, label="Anticipatory")
        ax.plot(post, label="Consummatory/Omission")
        idx = np.arange(len(onsets))
        if pos_mask.any():
            ax.plot(idx[pos_mask], post[pos_mask], ".", color="blue", ms=9, label=CS_INFO[cs][2])
        if (~pos_mask).any():
            ax.plot(idx[~pos_mask], post[~pos_mask], ".", color="red", ms=9, label=CS_INFO[cs][3])
        mean_anti = float(np.mean(anti)) if len(anti) else float("nan")
        ax.set_title("%s  antiLick %.2f" % (cs, mean_anti))
        ax.set_xlabel("trial #")
        if j == 0:
            ax.set_ylabel("Lick #")
            ax.legend(fontsize=8)
    return fig


def _beeswarm_offsets(values, width):
    """Symmetric x-offsets so equal (tied) values fan out instead of overlapping."""
    values = np.asarray(values, float)
    offsets = np.zeros(len(values))
    groups = {}
    for idx, v in enumerate(values):
        groups.setdefault(round(float(v)), []).append(idx)
    for idxs in groups.values():
        k = len(idxs)
        if k == 1:
            offsets[idxs[0]] = 0.0
        else:
            for pos, idx in zip(np.linspace(-width, width, k), idxs):
                offsets[idx] = pos
    return offsets


def _antilick_label(cs, cls):
    """X-axis label for a CS: name + empirical US-delivery rate (e.g. 'CS3\\n88%')."""
    pm = cls[cs]["pos_mask"]
    pct = (100.0 * pm.sum() / len(pm)) if len(pm) else 0.0
    return "%s\n%.0f%%" % (cs, pct)


def plot_anticipatory_lick_summary(
    df_events,
    paradigm,
    cls,
    meta,
    window_s=ANTILICK_WINDOW,
    summary=ANTILICK_SUMMARY,
    swarm_width=ANTILICK_SWARM_WIDTH,
    fig=None,
):
    """Beeswarm of anticipatory lick counts per CS (one dot = one trial).

    A ``mean +/- SD`` marker sits beside each cloud (``summary='median'`` ->
    ``median +/- IQR``). Draws into ``fig`` (a Figure or SubFigure) when given.
    """
    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamps"].to_numpy(float)
    names = paradigm["cs_list"]
    if len(licks) == 0 or not names:
        return None

    if fig is None:
        fig = plt.figure(figsize=(max(4.2, 1.6 * len(names) + 2.0), 3.6))
    ax = fig.subplots()

    tick_pos, tick_lab, all_counts, centers = [], [], [], []
    for i, cs in enumerate(names):
        counts = _anticipatory_lick_counts(licks, cls[cs]["onsets"], window_s)
        all_counts.append(counts)
        col = CS_INFO[cs][1]
        swarm_x, summ_x = i - 0.18, i + 0.22
        center = None
        if len(counts):
            dx = _beeswarm_offsets(counts.astype(float), swarm_width)
            ax.scatter(
                swarm_x + dx,
                counts,
                s=26,
                facecolor=col,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.75,
                zorder=2,
            )
            if summary == "median":
                center = float(np.median(counts))
                lo_e = float(np.percentile(counts, 25))
                hi_e = float(np.percentile(counts, 75))
            else:  # mean +/- SD
                center = float(np.mean(counts))
                sd = float(np.std(counts))
                lo_e, hi_e = center - sd, center + sd
            ax.plot(
                [summ_x, summ_x],
                [lo_e, hi_e],
                color="black",
                lw=3.2,
                solid_capstyle="round",
                zorder=3,
            )
            ax.plot(summ_x, center, "o", color="black", ms=7, zorder=4)
        centers.append(center)
        tick_pos.append(i)
        tick_lab.append(_antilick_label(cs, cls))

    ax.set_ylabel("Lick #")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ymax = max((c.max() for c in all_counts if len(c)), default=1)
    ax.set_ylim(-0.5, ymax * 1.08 + 1)
    # summary stat at the top of each column; only the first is prefixed with its name
    stat_label = "med" if summary == "median" else "avg"
    for i, center in enumerate(centers):
        if center is not None:
            txt = "%s %.1f" % (stat_label, center) if i == 0 else "%.1f" % center
            ax.text(i, ax.get_ylim()[1], txt, ha="center", va="top", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4)
    ax.set_title(
        "%s  %s   anticipatory lick (%.0f-%.0fs)"
        % (meta["subject_id"], meta["date"], window_s[0], window_s[1]),
        fontsize=9,
    )
    return fig


def plot_reaction_time(df_events, fig=None):
    """Reaction time from each reward to the next lick (seconds).

    Draws into ``fig`` (a Figure or SubFigure) when given, else creates one.
    """
    rew = df_events.loc[df_events["canonical"] == "Reward", "timestamps"].to_numpy(float)
    lick = np.sort(df_events.loc[df_events["canonical"] == "Lick", "timestamps"].to_numpy(float))
    if len(rew) == 0 or len(lick) == 0:
        return None
    rt = np.full(len(rew), np.nan)
    for i, r in enumerate(rew):
        nxt = lick[np.searchsorted(lick, r, side="left"):]
        if len(nxt):
            rt[i] = nxt[0] - r
    good = rt[~np.isnan(rt)]
    if fig is None:
        fig = plt.figure(figsize=(12, 4))
    ax = fig.subplots(1, 3)
    ax[0].plot(rt)
    ax[0].set_xlabel("Reward #")
    ax[0].set_ylabel("RT (s)")
    ax[0].set_title("median RT %.3f s" % (np.nanmedian(rt) if len(good) else float("nan")))
    if len(good):
        ax[1].hist(good)
        ax[2].hist(good[good < 0.5])
    ax[1].set_xlabel("RT (s)")
    ax[2].set_xlabel("RT < 0.5 s")
    return fig



def _resolve_want(plot_types):
    """Resolve the requested plot set into a concrete subset of names."""
    if plot_types is None or "all" in plot_types or "all_sess" in plot_types:
        return {"session", "psth", "lick", "antilick", "rt", "psth_compare_CS"}
    return set(plot_types)


def _summary_sections(
    df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw, antilick_kw, n_roi
):
    """Return ``[(height, draw_fn), ...]`` for each panel to place in the page.

    ``draw_fn`` takes a target SubFigure and renders one section into it.
    """
    licks = df_events.loc[df_events["canonical"] == "Lick", "timestamps"].to_numpy(float)
    rews = df_events.loc[df_events["canonical"] == "Reward", "timestamps"].to_numpy(float)
    row_h = 3.0 * max(n_roi, 1)
    sections = []

    if "session" in want:
        sections.append(
            (
                3.0 + 1.2 * max(n_roi, 1),
                lambda sf: plot_session_overview(
                    df_events, df_fip, paradigm, meta, channels, fig=sf
                ),
            )
        )
    # curation can drop every fiber; PSTH is FIP-only and subplots(0, 0) raises, whereas
    # the remaining sections are behavioral and stay valid with no photometry at all
    if "psth" in want and n_roi:
        for cs in paradigm["cs_list"]:
            sections.append(
                (
                    row_h + 1.2,
                    lambda sf, cs=cs: plot_cs_psth_grid(
                        df_fip, cs, cls, meta, channels, fig=sf, **psth_kw
                    ),
                )
            )
    if "lick" in want and len(licks) and len(paradigm["cs_list"]):
        sections.append((4.0, lambda sf: plot_lick_quant(df_events, paradigm, cls, fig=sf)))
    if "antilick" in want and len(licks) and len(paradigm["cs_list"]):
        sections.append(
            (
                4.0,
                lambda sf: plot_anticipatory_lick_summary(
                    df_events, paradigm, cls, meta, fig=sf, **antilick_kw
                ),
            )
        )
    if "rt" in want and len(rews) and len(licks):
        sections.append((4.0, lambda sf: plot_reaction_time(df_events, fig=sf)))
    if "psth_compare_CS" in want and n_roi and paradigm["cs_list"]:
        n_pairs = len(_channels_present(df_fip, channels))
        sections.append(
            (
                3.0 * max(n_pairs, 1) + 1.0,
                lambda sf: plot_cs_psth_compare(df_fip, paradigm, cls, meta, channels, fig=sf),
            )
        )
    return sections


def render_summary_figure(
    df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw, antilick_kw
):
    """Compose all requested panels into ONE tall figure and return it.

    Returns ``None`` if there is nothing to draw.
    """
    pairs = _channels_present(df_fip, channels)
    n_roi = len({r for _, r in pairs})
    n_chan = len({c for c, _ in pairs})

    sections = _summary_sections(
        df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw, antilick_kw, n_roi
    )
    if not sections:
        return None

    heights = [h for h, _ in sections]
    width = max(20.0, 4.5 * max(n_chan, 1))
    fig = plt.figure(figsize=(width, sum(heights)), layout="constrained")
    fig.suptitle(
        "%s  %s   [%s]" % (meta["subject_id"], meta["date"], paradigm["stage"]),
        fontsize=15,
        fontweight="bold",
    )

    subfigs = fig.subfigures(len(sections), 1, height_ratios=heights)
    if len(sections) == 1:
        subfigs = [subfigs]
    for (_, draw_fn), sf in zip(sections, subfigs):
        draw_fn(sf)
    return fig


def plot_nwb_summary(
    nwb,
    channels=None,
    meta=None,
    save_path=None,
    plot_types=None,
    t_before=T_BEFORE,
    t_after=T_AFTER,
    baseline=BASELINE,
    output_sampling_rate=OUTPUT_SR,
    antilick_window=ANTILICK_WINDOW,
    antilick_summary=ANTILICK_SUMMARY,
    antilick_swarm_width=ANTILICK_SWARM_WIDTH,
):
    """Render (and optionally save) the single-page summary for one session.

    The plotting half of :func:`analyze_nwb`. Takes the ``dummy_nwb`` and ``meta``
    from :func:`process_nwb` and re-derives the paradigm and trial classification from
    ``nwb.df_events``, so no analysis state has to be threaded through.

    Parameters
    ----------
    nwb : dummy_nwb
        Needs ``df_events`` and ``df_fip``; ``df_fip`` is drawn as given, so filter
        and curate before building the object.
    meta : dict
        Needs ``subject_id`` and ``date`` for the titles.
    channels : dict or None
        Optional ``{'<Chan>_<ROI>': 'location'}`` filter on the panels drawn.
    save_path : str or None
        If given, the page is written here as a PDF, and a PNG alongside it with the
        same stem (``.png``).
    plot_types : list or None
        ``['all_sess']``/``['all']`` -> everything, or a subset of
        ``{'session','psth','lick','antilick','rt','psth_compare_CS'}``.
    antilick_window : tuple
        ``(start, end)`` seconds after CS onset for the anticipatory-lick count.
    antilick_summary : str
        ``'mean'`` (mean +/- SD) or ``'median'`` (median +/- IQR).

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        ``None`` when there was nothing to draw. Closed (but still displayable) once
        saved, so a batch run does not accumulate open figures.
    paths : dict
        ``{'pdf': ..., 'png': ...}`` when saved, empty otherwise.
    """
    df_events, df_fip = nwb.df_events, nwb.df_fip
    if meta is None:
        subject_id, session_date = parse_session_name(nwb)
        meta = {"subject_id": subject_id, "date": session_date}
    paradigm = detect_paradigm(df_events)
    cls = classify_trials(df_events, paradigm["cs_list"])
    want = _resolve_want(plot_types)
    psth_kw = {
        "t_before": t_before,
        "t_after": t_after,
        "baseline": baseline,
        "output_sampling_rate": output_sampling_rate,
    }
    antilick_kw = {
        "window_s": antilick_window,
        "summary": antilick_summary,
        "swarm_width": antilick_swarm_width,
    }
    fig = render_summary_figure(
        df_events, df_fip, paradigm, cls, meta, channels, want, psth_kw, antilick_kw
    )

    paths = {}
    if fig is not None and save_path:
        pdf_path = save_path if save_path.endswith(".pdf") else save_path + ".pdf"
        png_path = pdf_path[:-4] + ".png"
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=PNG_DPI)
        plt.close(fig)
        paths = {"pdf": pdf_path, "png": png_path}
        print("[saved] %s" % pdf_path)
        print("[saved] %s" % png_path)
    return fig, paths



def _beeswarm_group(ax, x_pos, values, color, summary, swarm_width, alpha=0.70, summary_color="black"):
    """Draw one beeswarm column + mean/median summary marker at x_pos.

    Returns the summary center (mean or median), or None if there was no data.
    """
    values = np.asarray(values, float)
    valid = values[~np.isnan(values)]
    if not len(valid):
        return None
    dx = _beeswarm_offsets(valid, swarm_width)
    ax.scatter(x_pos - 0.18 + dx, valid, s=20, facecolor=color, edgecolor="white",
               linewidth=0.3, alpha=alpha, zorder=2)
    if summary == "median":
        center = float(np.median(valid))
        lo = float(np.percentile(valid, 25))
        hi = float(np.percentile(valid, 75))
    else:
        center = float(np.mean(valid))
        sd = float(np.std(valid))
        lo, hi = center - sd, center + sd
    ax.plot([x_pos + 0.22, x_pos + 0.22], [lo, hi], color=summary_color, lw=3.2,
            solid_capstyle="round", zorder=3)
    ax.plot(x_pos + 0.22, center, "o", color=summary_color, ms=7, zorder=4)
    return center


def psth_CS_fip(
    nwb,
    channels=None,
    meta=None,
    antilick_window=ANTILICK_WINDOW,
    summary=ANTILICK_SUMMARY,
    swarm_width=ANTILICK_SWARM_WIDTH,
    fig=None,
):
    """Per-fiber summary figure: beeswarm row + PSTH comparison row.

    For each fiber (unique event in df_fip), draws two rows:

    Row 1 — behavioral + scalar FIP:
      - Left panel  : anticipatory lick beeswarm per CS via
                      :func:`plot_anticipatory_lick_summary`
      - Right panel : CS response and reward response beeswarms (rewarded trials only),
                      using ``cs_response_<event>`` / ``rew_response_<event>`` columns
                      added by :func:`pavlovian_analysis.enrich_df_trials`

    Row 2 — PSTH comparison (:func:`plot_cs_psth_compare`) for that fiber only.

    Returns ``None`` if df_fip is empty.
    """
    df_fip = nwb.df_fip
    df_events = nwb.df_events
    df_trials = nwb.df_trials

    # channels should already be filtered
    # df_fip = _filter_channels(df_fip, channels) if channels else df_fip
    if df_fip is None or len(df_fip) == 0:
        return None

    if meta is None:
        subject_id, session_date = parse_session_name(nwb)
        meta = {"subject_id": subject_id, "date": session_date}

    events = list(df_fip["event"].unique())
    paradigm = detect_paradigm(df_events)
    cls = classify_trials(df_events, paradigm["cs_list"])
    cs_list = paradigm["cs_list"]
    n_cs = len(cs_list)
    n_fibers = len(events)

    if fig is None:
        fig = plt.figure(figsize=(18, 5.5 * n_fibers), layout="constrained")

    fiber_sfs = fig.subfigures(n_fibers, 1)
    if n_fibers == 1:
        fiber_sfs = [fiber_sfs]

    rewarded_df = df_trials[df_trials["rewarded"] == True] if df_trials is not None else None

    for sf, ev in zip(fiber_sfs, events):
        top_sf, bot_sf = sf.subfigures(2, 1, height_ratios=[1.2, 2.0])
        lick_sf, resp_sf = top_sf.subfigures(1, 2)

        # --- Left: reuse plot_anticipatory_lick_summary ---
        plot_anticipatory_lick_summary(
            df_events, paradigm, cls, meta,
            window_s=antilick_window, summary=summary,
            swarm_width=swarm_width, fig=lick_sf,
        )

        # --- Right: CS response + rew response (rewarded trials only) ---
        ax_resp = resp_sf.subplots()
        cs_col = "cs_response_%s" % ev
        rew_col = "rew_response_%s" % ev
        GAP = 1
        us_offset = n_cs + GAP

        if rewarded_df is not None and cs_col in df_trials.columns:
            cs_centers, rew_centers = [], []
            for j, cs in enumerate(cs_list):
                cs_df = rewarded_df[rewarded_df["CS_type"] == cs]
                color = CS_INFO[cs][1]
                cs_centers.append(_beeswarm_group(ax_resp, j,
                                cs_df[cs_col].dropna().to_numpy(float),
                                color, summary, swarm_width))
                rew_centers.append(_beeswarm_group(ax_resp, us_offset + j,
                                cs_df[rew_col].dropna().to_numpy(float),
                                color, summary, swarm_width))

            ax_resp.axhline(0, ls="--", color="gray", lw=0.8)
            ax_resp.axvline(n_cs - 0.5 + GAP / 2, ls=":", color="gray", lw=1, alpha=0.5)
            xlim = (-0.6, us_offset + n_cs - 0.4)
            ax_resp.set_xlim(*xlim)
            ax_resp.set_xticks(list(range(n_cs)) + [us_offset + j for j in range(n_cs)])
            ax_resp.set_xticklabels([cs for cs in cs_list] + [cs for cs in cs_list], fontsize=8)
            ax_resp.set_ylabel("%s\nΔF/F (%%)" % ev)

            # summary stat above each column; only the first of each group is prefixed
            stat_label = "med" if summary == "median" else "avg"
            ymax = ax_resp.get_ylim()[1]
            for j, center in enumerate(cs_centers):
                if center is not None:
                    txt = "%s %.1f" % (stat_label, center) if j == 0 else "%.1f" % center
                    ax_resp.text(j, ymax, txt, ha="center", va="top", fontsize=7)
            for j, center in enumerate(rew_centers):
                if center is not None:
                    txt = "%s %.1f" % (stat_label, center) if j == 0 else "%.1f" % center
                    ax_resp.text(us_offset + j, ymax, txt, ha="center", va="top", fontsize=7)

            x_range = xlim[1] - xlim[0]
            ax_resp.text(((n_cs - 1) / 2 - xlim[0]) / x_range, 1.03,
                         "CS response", transform=ax_resp.transAxes,
                         ha="center", va="bottom", fontsize=8)
            ax_resp.text((us_offset + (n_cs - 1) / 2 - xlim[0]) / x_range, 1.03,
                         "Rew response", transform=ax_resp.transAxes,
                         ha="center", va="bottom", fontsize=8)
        ax_resp.spines["top"].set_visible(False)
        ax_resp.spines["right"].set_visible(False)

        # --- Bottom: PSTH comparison for this fiber only ---
        plot_cs_psth_compare(df_fip[df_fip["event"] == ev], paradigm, cls, meta, fig=bot_sf)

    return fig
