from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..analysis.summariser._summary import ContactSummary

import altair as alt
import numpy as np
import pandas as pd

from ..utils import AgeGroupSpecs
from ._utils import _default_style, _merge_style, generate_vega_expression, ravel_matrix


def plot_mosaic(
    matrix: np.ndarray,
    title: str = "Contact pattern",
    xlabel: str = "Age of contacting individual",
    ylabel: str = "Age of contacted individual",
    zlabel: str = None,
    width: int | float = 250,
    height: int | float = 250,
    color_scheme: str = "spectral",
    color_reverse: bool = True,
    color_min: float = None,
    color_max: float = None,
    color_mid: float = None,
    x_tick_values: list = None,
    x_tick_labels: list = None,
    y_tick_values: list = None,
    y_tick_labels: list = None,
    z_tick_values: list = None,
    legend_position: str = "right",
    show_x_ticks: bool = True,
    show_x_label: bool = True,
    show_y_ticks: bool = True,
    show_y_label: bool = True,
    style_config: Optional[dict] = None,
) -> alt.Chart:
    """
    Plot a mosaic visualization of a contact matrix.

    Parameters
    ----------
    matrix : np.ndarray
        A 2D array representing the contact intensity or rate matrix.
    title : str, optional
        The title of the chart. Default is 'Contact pattern'.
    xlabel : str, optional
        Label for the x-axis. Default is 'Age of contacting individual'.
    ylabel : str, optional
        Label for the y-axis. Default is 'Age of contacted individual'.
    zlabel : str, optional
        Label for the color scale legend. If None, the legend is not displayed.
    width : int, optional
        The width of the chart in pixels. Default is 250.
    height : int, optional
        The height of the chart in pixels. Default is 250.
    color_scheme : str, optional
        The name of the color scheme to use. Default is 'spectral'.
    color_reverse : bool, optional
        Whether to reverse the color scheme. Default is True.
    color_min : float, optional
        Minimum value for the color scale domain. If None, uses the data minimum.
    color_max : float, optional
        Maximum value for the color scale domain. If None, uses the data maximum.
    x_tick_values : list, optional
        List of values where ticks should be placed on the x-axis. If None, defaults to
        every 10th value starting from 0.
    x_tick_labels : list, optional
        List of labels corresponding to x_tick_values. If None, uses x_tick_values as labels.
        Must be the same length as x_tick_values if provided.
    y_tick_values : list, optional
        List of values where ticks should be placed on the y-axis. If None, defaults to
        every 10th value starting from 0.
    y_tick_labels : list, optional
        List of labels corresponding to y_tick_values. If None, uses y_tick_values as labels.
        Must be the same length as y_tick_values if provided.
    z_tick_values : list, optional
        List of values where ticks should be placed on the color scale legend.
    legend_position : str, optional
        Position of the legend. Valid values include 'left', 'right', 'top', 'bottom',
        'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'. Default is 'right'.
    show_x_ticks : bool, optional
        Whether to show x-axis tick marks and tick labels. Default is True.
    show_x_label : bool, optional
        Whether to show the x-axis title. Default is True.
    show_y_ticks : bool, optional
        Whether to show y-axis tick marks and tick labels. Default is True.
    show_y_label : bool, optional
        Whether to show the y-axis title. Default is True.
    style_config : dict, optional
        A dictionary to override default style settings for axes, title, and legend.
        The keys can include 'x_axis', 'y_axis', 'title', 'legend', and 'color_scale'.

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the mosaic visualisation of a contact matrix.
    """
    alt.data_transformers.disable_max_rows()

    default_config = _default_style(label_angle_x=0, legend_position=legend_position)
    default_config["color_scale"] = {"scheme": color_scheme, "reverse": color_reverse}
    _merge_style(default_config, style_config)
    if color_mid is not None:
        default_config["color_scale"]["domainMid"] = color_mid
    if color_min is not None and color_max is not None:
        default_config["color_scale"]["domain"] = [color_min, color_max]

    if z_tick_values is not None:
        default_config["legend"]["values"] = z_tick_values

    x_indices, y_indices, values = ravel_matrix(matrix)
    color_values = values
    source = pd.DataFrame(
        {"x": x_indices, "y": y_indices, "z": color_values, "z_orig": values}
    )

    # Configure x-axis tick values and labels
    if x_tick_values is None:
        x_tick_values = list(range(0, matrix.shape[0], 10))

    x_tick_config = {"values": x_tick_values}
    if x_tick_labels is not None:
        if len(x_tick_labels) != len(x_tick_values):
            raise ValueError("x_tick_labels must have the same length as x_tick_values")
        x_tick_config["labelExpr"] = generate_vega_expression(
            x_tick_values, x_tick_labels
        )

    # Configure y-axis tick values and labels
    if y_tick_values is None:
        y_tick_values = list(range(0, matrix.shape[0], 10))

    y_tick_config = {"values": y_tick_values}
    if y_tick_labels is not None:
        if len(y_tick_labels) != len(y_tick_values):
            raise ValueError("y_tick_labels must have the same length as y_tick_values")
        y_tick_config["labelExpr"] = generate_vega_expression(
            y_tick_values, y_tick_labels
        )

    x_axis = alt.Axis(
        **x_tick_config,
        **default_config["x_axis"],
        labels=show_x_ticks,
        ticks=show_x_ticks,
    )

    y_axis = alt.Axis(
        **y_tick_config,
        **default_config["y_axis"],
        labels=show_y_ticks,
        ticks=show_y_ticks,
    )

    chart = (
        alt.Chart(source)
        .mark_rect(stroke=None)
        .encode(
            x=alt.X(
                "x:O",
                axis=x_axis,
                title=xlabel if show_x_label else None,
            ),
            y=alt.Y(
                "y:O",
                scale=alt.Scale(reverse=True),
                axis=y_axis,
                title=ylabel if show_y_label else None,
            ),
            color=alt.Color(
                "z:Q",
                scale=alt.Scale(**default_config["color_scale"]),
                title=zlabel,
                legend=alt.Legend(**default_config["legend"]) if zlabel else None,
            ),
        )
        .properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, **default_config["title"]),
        )
    )

    return chart


def plot_mosaic_pixilated(
    matrix: "np.ndarray | ContactSummary",
    age_group_specs: AgeGroupSpecs | None = None,
    title: str = "Contact pattern",
    xlabel: str = "Age of contacting individual",
    ylabel: str = "Age of contacted individual",
    zlabel: str = None,
    width: int | float = 250,
    height: int | float = 250,
    color_scheme: str = "spectral",
    color_reverse: bool = True,
    color_min: Optional[float] = None,
    color_max: Optional[float] = None,
    color_mid: Optional[float] = None,
    show_x_ticks: bool = True,
    show_x_label: bool = True,
    show_y_ticks: bool = True,
    show_y_label: bool = True,
    style_config: dict = None,
) -> alt.Chart:
    """Plot a pixilated (coarse-age) mosaic contact matrix.

    Parameters
    ----------
    matrix : np.ndarray or ContactSummary
        The coarse-age contact matrix to visualise. If a ContactSummary is passed,
        ``central`` is plotted and ``age_group_specs`` is inferred from it (unless
        overridden by the explicit ``age_group_specs`` argument).
    age_group_specs : AgeGroupSpecs, optional
        Required when *matrix* is an ``np.ndarray``. Optional when *matrix* is a
        ContactSummary that already carries its own ``age_group_specs``.
    title : str, optional
        Chart title. Default is 'Contact pattern'.
    xlabel, ylabel : str, optional
        Axis labels.
    zlabel : str, optional
        Color-scale legend title. Legend is hidden when None.
    width, height : int or float, optional
        Chart dimensions in pixels. Default 250 × 250.
    color_scheme : str, optional
        Vega color scheme name. Default is ``'spectral'``.
    color_reverse : bool, optional
        Whether to reverse the color scheme. Default is True.
    color_min, color_max : float, optional
        Fixed domain bounds for the color scale. Both must be set to take effect.
    color_mid : float, optional
        Midpoint for the color scale. Useful for diverging color schemes.
    show_x_ticks : bool, optional
        Whether to show x-axis tick marks and tick labels. Default is True.
    show_x_label : bool, optional
        Whether to show the x-axis title. Default is True.
    show_y_ticks : bool, optional
        Whether to show y-axis tick marks and tick labels. Default is True.
    show_y_label : bool, optional
        Whether to show the y-axis title. Default is True.
    style_config : dict, optional
        Dict of Altair/Vega-Lite overrides keyed by section
        (``'x_axis'``, ``'y_axis'``, ``'title'``, ``'legend'``).
    """
    alt.data_transformers.disable_max_rows()

    # Resolve ContactSummary input
    from ..analysis.summariser._summary import (
        ContactSummary,
    )  # lazy to avoid circular import

    if isinstance(matrix, ContactSummary):
        resolved_specs = age_group_specs or matrix.age_group_specs
        if resolved_specs is None:
            raise ValueError(
                "ContactSummary has no age_group_specs; pass age_group_specs explicitly."
            )
        age_group_specs = resolved_specs
        matrix = matrix.central
    elif age_group_specs is None:
        raise ValueError("age_group_specs is required when matrix is an np.ndarray.")

    default_config = _default_style(label_angle_x=-45)
    _merge_style(default_config, style_config)

    # Tile each coarse-bin value into its single-year block so the plotted
    # color reflects the actual intensity. Bin width is encoded visually by
    # block size, not by rescaling the value.
    expanded_matrix = np.zeros((age_group_specs.range, age_group_specs.range))
    for i in range(len(age_group_specs.left)):
        x0, x1 = age_group_specs.left[i], age_group_specs.right[i] + 1
        for j in range(len(age_group_specs.left)):
            y0, y1 = age_group_specs.left[j], age_group_specs.right[j] + 1
            expanded_matrix[x0:x1, y0:y1] = matrix[i, j]

    x_indices, y_indices, values = ravel_matrix(expanded_matrix)
    source = pd.DataFrame({"x": x_indices, "y": y_indices, "z": values})

    tick_pos = [
        np.floor(np.mean([age_group_specs.left[i], age_group_specs.right[i] + 1]))
        for i in range(len(age_group_specs.left))
    ]
    tick_labels = [
        f"[{age_group_specs.left[i]},{age_group_specs.right[i] + 1})"
        for i in range(len(age_group_specs.left))
    ]
    expression = generate_vega_expression(tick_pos, tick_labels)

    default_config["color_scale"] = {"scheme": color_scheme, "reverse": color_reverse}
    _merge_style(default_config, style_config)
    if color_mid is not None:
        default_config["color_scale"]["domainMid"] = color_mid
    if color_min is not None and color_max is not None:
        default_config["color_scale"]["domain"] = [color_min, color_max]

    chart = (
        alt.Chart(source)
        .mark_rect()
        .encode(
            x=alt.X(
                "x:O",
                axis=alt.Axis(
                    values=tick_pos,
                    labelExpr=expression,
                    **default_config["x_axis"],
                    labels=show_x_ticks,
                    ticks=show_x_ticks,
                ),
                title=xlabel if show_x_label else None,
            ),
            y=alt.Y(
                "y:O",
                scale=alt.Scale(reverse=True),
                axis=alt.Axis(
                    values=tick_pos,
                    labelExpr=expression,
                    **default_config["y_axis"],
                    labels=show_y_ticks,
                    ticks=show_y_ticks,
                ),
                title=ylabel if show_y_label else None,
            ),
            color=alt.Color(
                "z:Q",
                scale=alt.Scale(**default_config["color_scale"]),
                title=zlabel,
                legend=alt.Legend(**default_config["legend"]) if zlabel else None,
            ),
        )
        .properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, **default_config["title"]),
        )
    )

    return chart


def plot_mosaic_marginal(
    mcint: np.ndarray,
    mcint_lb: np.ndarray = None,
    mcint_ub: np.ndarray = None,
    width: int = 250,
    height: int = 250,
    title: str = "Contact intensity",
    style_config: Optional[dict] = None,
) -> alt.Chart:
    """Plot the marginal contact intensity with optional uncertainty bands.

    Parameters
           ----------
           mcint : np.ndarray
                   Array representing the main contact intensity values.
           mcint_lb : np.ndarray, optional
                   Array representing the lower bound of the uncertainty band. If provided, both mcint_lb and mcint_ub are used to
                   display an error band around the line plot. Default is None.
           mcint_ub : np.ndarray, optional
                   Array representing the upper bound of the uncertainty band. If provided, both mcint_lb and mcint_ub are used to
                   display an error band around the line plot. Default is None.
           width : int, optional
                   The width of the resulting chart in pixels. Default is 250.
           height : int, optional
                   The height of the resulting chart in pixels. Default is 250.
           title : str, optional
                   The title for the chart. Default is 'Contact intensity'.
           style_config : dict, optional
                   A dictionary for overriding default style configurations for the axes and title. The keys should correspond to the
                   configuration parts ('x_axis', 'y_axis', or 'title') and the values should be dictionaries of style parameters.
                   Default is None.

           Returns
           -------
           alt.Chart
                   An Altair Chart object that visualizes the marginal contact intensity. When error bounds are provided, the chart includes
                   an error band alongside the main line plot.
    """
    alt.data_transformers.disable_max_rows()

    config = _default_style()
    config["x_axis"].update({"values": list(range(0, 100, 10)), "grid": True})
    config["y_axis"].update({"values": list(range(0, 100, 5)), "grid": True})
    _merge_style(config, style_config)

    df = pd.DataFrame({"x": np.arange(mcint.size), "y": mcint})
    has_band = mcint_lb is not None and mcint_ub is not None
    if has_band:
        df["l"] = mcint_lb
        df["u"] = mcint_ub

    x_axis = alt.Axis(title="Age of contacting individuals", **config["x_axis"])
    y_axis = alt.Axis(title="Contact intensity", **config["y_axis"])

    base = alt.Chart(df).encode(
        x=alt.X("x:O", axis=x_axis), y=alt.Y("y:Q", axis=y_axis)
    )
    line = base.mark_line()

    if has_band:
        band = (
            alt.Chart(df)
            .mark_errorband()
            .encode(x=alt.X("x:O", axis=x_axis), y=alt.Y("l:Q", axis=y_axis), y2="u:Q")
        )
        chart = band + line
    else:
        chart = line

    return chart.properties(
        width=width, height=height, title=alt.TitleParams(text=title, **config["title"])
    )
