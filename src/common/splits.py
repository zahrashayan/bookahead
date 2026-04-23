"""Shared dataset splitting helpers."""

from __future__ import annotations


def grouped_time_split(df, panel_col="panel", time_col="scraped_date", train_frac=0.8):
    """Split chronologically by panel start time to reduce leakage."""
    panel_first = df.groupby(panel_col)[time_col].min().sort_values()
    panels = panel_first.index.to_list()
    cut = int(len(panels) * train_frac)
    train_panels = set(panels[:cut])
    train = df[df[panel_col].isin(train_panels)].copy()
    test = df[~df[panel_col].isin(train_panels)].copy()
    return train, test
