"""
SPN Quality Dashboard Generator.

Creates comprehensive visualizations, summary statistics, and HTML reports
for evaluating SPN quality across local and global models.
"""

import logging
import os
from pathlib import Path

import numpy as np

# Try imports
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Dashboard disabled.")


# ============================================================================
# Quality Thresholds and Ratings
# ============================================================================

THRESHOLDS = {
    "train_ll": {
        "good": -8.0,  # Better than -8
        "fair": -12.0,  # Between -12 and -8
        # Poor: Worse than -12
    },
    "mmd_pvalue": {
        "good": 0.05,  # p > 0.05 (not significant)
        "fair": 0.01,  # 0.01 < p < 0.05
        # Poor: p < 0.01
    },
    "ks_fail_ratio": {
        "good": 0.3,  # < 30% failed
        "fair": 0.5,  # 30-50% failed
        # Poor: > 50% failed
    },
    "overall_accuracy": {
        "good": 0.75,  # > 75%
        "fair": 0.60,  # 60-75%
        # Poor: < 60%
    },
    "skeleton_accuracy": {
        "good": 0.80,  # > 80%
        "fair": 0.65,  # 65-80%
        # Poor: < 65%
    },
    "overall_f1": {
        "good": 0.60,  # > 0.60
        "fair": 0.40,  # 0.40-0.60
        # Poor: < 0.40
    },
}


def get_quality_rating(metric_name, value):
    """
    Get quality rating (Good/Fair/Poor) for a metric value.

    Args:
        metric_name: Name of metric (e.g., 'train_ll')
        value: Metric value

    Returns:
        rating: 'Good', 'Fair', or 'Poor'
        color: Color code for visualization
    """
    if value is None:
        return "N/A", "gray"

    thresholds = THRESHOLDS.get(metric_name, {})

    # Handle different threshold directions
    if metric_name in ["train_ll"]:
        # Higher is better
        if value >= thresholds.get("good", float("inf")):
            return "Good", "#2ecc71"  # Green
        elif value >= thresholds.get("fair", float("inf")):
            return "Fair", "#f39c12"  # Orange
        else:
            return "Poor", "#e74c3c"  # Red
    elif metric_name in ["mmd_pvalue"]:
        # Higher is better (want non-significant = good fit)
        if value >= thresholds.get("good", float("inf")):
            return "Good", "#2ecc71"
        elif value >= thresholds.get("fair", float("inf")):
            return "Fair", "#f39c12"
        else:
            return "Poor", "#e74c3c"
    else:
        # Default: lower thresholds mean higher is better
        if value >= thresholds.get("good", float("inf")):
            return "Good", "#2ecc71"
        elif value >= thresholds.get("fair", float("inf")):
            return "Fair", "#f39c12"
        else:
            return "Poor", "#e74c3c"


# ============================================================================
# Summary Statistics
# ============================================================================


def compute_summary_statistics(local_results):
    """
    Compute summary statistics across local SPNs.

    Args:
        local_results: List of result dictionaries from evaluate_spn_quality

    Returns:
        summary: Dictionary with mean/std for each metric
    """
    summary = {}

    if not local_results:
        return summary

    # Metrics to aggregate
    metrics = [
        "train_ll",
        "mmd_squared",
        "mmd_pvalue",
        "ks_fail_ratio",
        "overall_accuracy",
        "overall_f1",
        "skeleton_accuracy",
    ]

    for metric in metrics:
        values = [r.get(metric) for r in local_results if r.get(metric) is not None]
        if values:
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_min"] = np.min(values)
            summary[f"{metric}_max"] = np.max(values)

    return summary


# ============================================================================
# Dashboard Plots
# ============================================================================


def create_summary_dashboard(
    local_results, global_result, summary_stats, save_path, config_info=None
):
    """
    Create comprehensive 4-panel dashboard plot.

    Args:
        local_results: List of local SPN evaluation results
        global_result: Global SPN evaluation result
        summary_stats: Summary statistics dictionary
        save_path: Path to save dashboard PNG
        config_info: Optional dict with run configuration

    Returns:
        Path to saved plot
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # Title
        if config_info:
            scenario = config_info.get("scenario", "unknown")
            K = config_info.get("K", "?")
            d = config_info.get("d", "?")
            n = config_info.get("n", "?")
            fig.suptitle(
                f"SPN Quality Dashboard: {scenario.upper()} | K={K} clients, d={d} features, n={n} samples",
                fontsize=16,
                fontweight="bold",
            )
        else:
            fig.suptitle("SPN Quality Dashboard", fontsize=16, fontweight="bold")

        # Panel 1: Train Log-Likelihood
        ax1 = fig.add_subplot(gs[0, 0])
        plot_train_ll_comparison(ax1, local_results, global_result)

        # Panel 2: MMD & KS Tests
        ax2 = fig.add_subplot(gs[0, 1])
        plot_distribution_tests(ax2, local_results, global_result)

        # Panel 3: CI Accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        plot_ci_accuracy(ax3, local_results, global_result)

        # Panel 4: Quality Ratings
        ax4 = fig.add_subplot(gs[1, :])
        plot_quality_ratings(ax4, local_results, global_result)

        # Panel 5: Summary Statistics Table
        ax5 = fig.add_subplot(gs[2, :])
        plot_summary_table(ax5, summary_stats, len(local_results))

        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()

        logging.info(f"  Saved dashboard: {save_path}")
        return save_path

    except Exception as e:
        logging.warning(f"Failed to create dashboard: {e}")
        return None


def plot_train_ll_comparison(ax, local_results, global_result):
    """Plot train log-likelihood comparison."""
    # Extract values
    local_lls = [r.get("train_ll") for r in local_results if r.get("train_ll")]
    names = [f"Client {i}" for i in range(len(local_lls))]

    if global_result and global_result.get("train_ll"):
        local_lls.append(global_result["train_ll"])
        names.append("Global")

    if not local_lls:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("Train Log-Likelihood")
        return

    # Color bars by quality
    colors = []
    for ll in local_lls:
        _, color = get_quality_rating("train_ll", ll)
        colors.append(color)

    # Plot
    bars = ax.barh(names, local_lls, color=colors, alpha=0.7, edgecolor="black")
    ax.axvline(THRESHOLDS["train_ll"]["good"], color="green", linestyle="--", alpha=0.5)
    ax.axvline(
        THRESHOLDS["train_ll"]["fair"], color="orange", linestyle="--", alpha=0.5
    )

    ax.set_xlabel("Train LL (higher is better)", fontsize=10)
    ax.set_title("Train Log-Likelihood", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, local_lls)):
        ax.text(val, i, f" {val:.2f}", va="center", fontsize=9)


def plot_distribution_tests(ax, local_results, global_result):
    """Plot MMD and KS test results."""
    # Combine local + global
    all_results = local_results + ([global_result] if global_result else [])
    names = [f"C{i}" for i in range(len(local_results))] + (
        ["Global"] if global_result else []
    )

    mmd_pvals = [r.get("mmd_pvalue") for r in all_results]
    ks_fails = [r.get("ks_fail_ratio") for r in all_results]

    if not mmd_pvals and not ks_fails:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("Distribution Quality Tests")
        return

    x = np.arange(len(names))
    width = 0.35

    # Plot MMD p-values
    if any(v is not None for v in mmd_pvals):
        mmd_vals = [v if v is not None else 0 for v in mmd_pvals]
        ax.bar(
            x - width / 2,
            mmd_vals,
            width,
            label="MMD p-value",
            color="#3498db",
            alpha=0.7,
        )

    # Plot KS fail ratio (inverted for better visual)
    if any(v is not None for v in ks_fails):
        ks_pass = [1 - v if v is not None else 0 for v in ks_fails]
        ax.bar(
            x + width / 2,
            ks_pass,
            width,
            label="KS pass ratio",
            color="#9b59b6",
            alpha=0.7,
        )

    # Threshold lines
    ax.axhline(0.05, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Distribution Quality Tests", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.1])


def plot_ci_accuracy(ax, local_results, global_result):
    """Plot conditional independence accuracy."""
    # Combine local + global
    all_results = local_results + ([global_result] if global_result else [])
    names = [f"C{i}" for i in range(len(local_results))] + (
        ["Global"] if global_result else []
    )

    overall_acc = [r.get("overall_accuracy") for r in all_results]
    skeleton_acc = [r.get("skeleton_accuracy") for r in all_results]

    if not overall_acc and not skeleton_acc:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("CI Test Accuracy")
        return

    x = np.arange(len(names))
    width = 0.35

    # Plot overall accuracy
    if any(v is not None for v in overall_acc):
        overall_vals = [v if v is not None else 0 for v in overall_acc]
        ax.bar(
            x - width / 2,
            overall_vals,
            width,
            label="Overall Acc",
            color="#e74c3c",
            alpha=0.7,
        )

    # Plot skeleton accuracy
    if any(v is not None for v in skeleton_acc):
        skel_vals = [v if v is not None else 0 for v in skeleton_acc]
        ax.bar(
            x + width / 2,
            skel_vals,
            width,
            label="Skeleton Acc",
            color="#2ecc71",
            alpha=0.7,
        )

    # Threshold lines
    ax.axhline(0.75, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(0.60, color="orange", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_ylabel("Accuracy")
    ax.set_title("CI Test Accuracy", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.1])


def plot_quality_ratings(ax, local_results, global_result):
    """Plot quality rating heatmap."""
    ax.axis("off")

    metrics = [
        ("train_ll", "Train LL"),
        ("mmd_pvalue", "MMD p-val"),
        ("ks_fail_ratio", "KS pass"),
        ("overall_accuracy", "CI Acc"),
        ("skeleton_accuracy", "Skel Acc"),
        ("overall_f1", "CI F1"),
    ]

    # Prepare data
    all_results = local_results + ([global_result] if global_result else [])
    names = ["Client " + str(i) for i in range(len(local_results))] + (
        ["Global"] if global_result else []
    )

    # Create rating grid
    n_models = len(all_results)
    n_metrics = len(metrics)

    # Title
    ax.text(
        0.5,
        0.95,
        "Quality Ratings",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Grid
    cell_height = 0.8 / (n_models + 1)
    cell_width = 0.9 / n_metrics

    # Header row
    for j, (_, label) in enumerate(metrics):
        x = 0.05 + j * cell_width
        y = 0.85
        ax.text(
            x + cell_width / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Data rows
    for i, (result, name) in enumerate(zip(all_results, names)):
        # Model name
        ax.text(
            0.02, 0.85 - (i + 1) * cell_height, name, ha="left", va="center", fontsize=9
        )

        # Metrics
        for j, (metric_key, _) in enumerate(metrics):
            x = 0.05 + j * cell_width
            y = 0.85 - (i + 1) * cell_height

            value = result.get(metric_key)

            # Special handling for KS (invert)
            if metric_key == "ks_fail_ratio" and value is not None:
                value = 1 - value  # Show pass ratio instead

            rating, color = get_quality_rating(metric_key, value)

            # Draw cell
            rect = plt.Rectangle(
                (x, y - cell_height / 2),
                cell_width,
                cell_height,
                facecolor=color,
                alpha=0.3,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)

            # Rating text
            ax.text(
                x + cell_width / 2,
                y,
                rating,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )


def plot_summary_table(ax, summary_stats, n_local):
    """Plot summary statistics table."""
    ax.axis("off")

    if not summary_stats:
        ax.text(0.5, 0.5, "No summary statistics", ha="center", va="center")
        return

    ax.text(
        0.5,
        0.95,
        f"Summary Statistics (across {n_local} local SPNs)",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Prepare table data
    table_data = []
    metrics = [
        ("train_ll", "Train LL"),
        ("mmd_pvalue", "MMD p-value"),
        ("ks_fail_ratio", "KS fail ratio"),
        ("overall_accuracy", "CI Accuracy"),
        ("skeleton_accuracy", "Skeleton Acc"),
        ("overall_f1", "CI F1"),
    ]

    for metric_key, label in metrics:
        mean_key = f"{metric_key}_mean"
        std_key = f"{metric_key}_std"

        if mean_key in summary_stats:
            mean_val = summary_stats[mean_key]
            std_val = summary_stats.get(std_key, 0)
            rating, _ = get_quality_rating(metric_key, mean_val)

            table_data.append([label, f"{mean_val:.3f}", f"{std_val:.3f}", rating])

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=["Metric", "Mean", "Std Dev", "Rating"],
            cellLoc="center",
            loc="center",
            bbox=[0.1, 0.1, 0.8, 0.7],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Color header
        for i in range(4):
            table[(0, i)].set_facecolor("#34495e")
            table[(0, i)].set_text_props(color="white", weight="bold")

        # Color rating column
        for i, row in enumerate(table_data):
            rating = row[3]
            if rating == "Good":
                table[(i + 1, 3)].set_facecolor("#2ecc71")
            elif rating == "Fair":
                table[(i + 1, 3)].set_facecolor("#f39c12")
            elif rating == "Poor":
                table[(i + 1, 3)].set_facecolor("#e74c3c")


# ============================================================================
# HTML Report Generation
# ============================================================================


def generate_html_report(
    local_results,
    global_result,
    summary_stats,
    config_info,
    output_dir,
    dashboard_path=None,
    umap_paths=None,
):
    """
    Generate comprehensive HTML report with all metrics and visualizations.

    Args:
        local_results: List of local SPN evaluation results
        global_result: Global SPN evaluation result
        summary_stats: Summary statistics dictionary
        config_info: Run configuration dictionary
        output_dir: Directory to save report
        dashboard_path: Path to dashboard PNG (relative to output_dir)
        umap_paths: Dictionary of UMAP plot paths

    Returns:
        Path to HTML report
    """
    report_path = Path(output_dir) / "spn_quality_report.html"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SPN Quality Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .config-info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .config-info table {{
            width: 100%;
        }}
        .config-info td {{
            padding: 5px;
        }}
        .config-info td:first-child {{
            font-weight: bold;
            width: 200px;
        }}
        .dashboard {{
            text-align: center;
            margin: 30px 0;
        }}
        .dashboard img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .metric-card h4 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .rating-good {{
            background-color: #2ecc71;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .rating-fair {{
            background-color: #f39c12;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .rating-poor {{
            background-color: #e74c3c;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .umap-section {{
            margin: 30px 0;
        }}
        .umap-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .umap-grid img {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 SPN Quality Evaluation Report</h1>

        <div class="config-info">
            <h3>Run Configuration</h3>
            <table>
                <tr><td>Scenario:</td><td>{config_info.get('scenario', 'N/A')}</td></tr>
                <tr><td>Number of Clients (K):</td><td>{config_info.get('K', 'N/A')}</td></tr>
                <tr><td>Number of Features (d):</td><td>{config_info.get('d', 'N/A')}</td></tr>
                <tr><td>Total Samples (n):</td><td>{config_info.get('n', 'N/A')}</td></tr>
                <tr><td>Model Type:</td><td>{config_info.get('model_type', 'N/A')}</td></tr>
                <tr><td>CI Method:</td><td>{config_info.get('ci_method', 'N/A')}</td></tr>
                <tr><td>Alpha:</td><td>{config_info.get('alpha', 'N/A')}</td></tr>
                <tr><td>Device:</td><td>{config_info.get('device', 'N/A')}</td></tr>
                <tr><td>SPN Training Epochs:</td><td>{config_info.get('epochs', 'N/A')}</td></tr>
            </table>
        </div>

        <h2>📊 Dashboard Overview</h2>
        <div class="dashboard">
            {f'<img src="{Path(dashboard_path).name}" alt="Dashboard">' if dashboard_path else '<p>Dashboard not available</p>'}
        </div>

        <h2>📈 Summary Statistics</h2>
        {_generate_summary_html(summary_stats, len(local_results))}

        <h2>🖥️ Local SPNs (Individual Clients)</h2>
        {_generate_local_results_html(local_results)}

        <h2>🌐 Global Federated SPN</h2>
        {_generate_global_result_html(global_result)}

        <h2>🗺️ UMAP Visualizations</h2>
        <div class="umap-section">
            {_generate_umap_section(umap_paths, output_dir) if umap_paths else '<p>No UMAP visualizations available</p>'}
        </div>

        <div class="footer">
            <p>Generated by FedCDH SPN Quality Dashboard</p>
            <p>Output Directory: {output_dir}</p>
        </div>
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(report_path, "w") as f:
        f.write(html_content)

    logging.info(f"  Saved HTML report: {report_path}")
    return str(report_path)


def _generate_summary_html(summary_stats, n_local):
    """Generate HTML for summary statistics section."""
    if not summary_stats:
        return "<p>No summary statistics available</p>"

    metrics = [
        ("train_ll", "Train Log-Likelihood"),
        ("mmd_pvalue", "MMD p-value"),
        ("ks_fail_ratio", "KS Failure Ratio"),
        ("overall_accuracy", "CI Accuracy"),
        ("skeleton_accuracy", "Skeleton Accuracy"),
        ("overall_f1", "CI F1 Score"),
    ]

    html = f"<p><strong>Statistics computed across {n_local} local SPNs</strong></p>"
    html += "<table>"
    html += "<tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Rating</th></tr>"

    for metric_key, label in metrics:
        mean_key = f"{metric_key}_mean"
        if mean_key in summary_stats:
            mean_val = summary_stats[mean_key]
            std_val = summary_stats.get(f"{metric_key}_std", 0)
            min_val = summary_stats.get(f"{metric_key}_min", 0)
            max_val = summary_stats.get(f"{metric_key}_max", 0)

            rating, _ = get_quality_rating(metric_key, mean_val)
            rating_class = f"rating-{rating.lower()}"

            html += f"<tr>"
            html += f"<td>{label}</td>"
            html += f"<td>{mean_val:.3f}</td>"
            html += f"<td>{std_val:.3f}</td>"
            html += f"<td>{min_val:.3f}</td>"
            html += f"<td>{max_val:.3f}</td>"
            html += f'<td><span class="{rating_class}">{rating}</span></td>'
            html += f"</tr>"

    html += "</table>"
    return html


def _generate_local_results_html(local_results):
    """Generate HTML for local SPN results."""
    if not local_results:
        return "<p>No local SPN results available</p>"

    html = ""
    for i, result in enumerate(local_results):
        html += f'<div class="metric-card">'
        html += f"<h4>Client {i}</h4>"
        html += _generate_single_result_html(result)
        html += "</div>"

    return html


def _generate_global_result_html(global_result):
    """Generate HTML for global SPN result."""
    if not global_result:
        return "<p>No global SPN result available</p>"

    html = '<div class="metric-card">'
    html += _generate_single_result_html(global_result)
    html += "</div>"
    return html


def _generate_single_result_html(result):
    """Generate HTML for a single SPN result."""
    html = "<table>"

    metrics = [
        ("train_ll", "Train Log-Likelihood", "train_ll"),
        ("mmd_squared", "MMD²", None),
        ("mmd_pvalue", "MMD p-value", "mmd_pvalue"),
        ("ks_fail_ratio", "KS Failure Ratio", "ks_fail_ratio"),
        ("overall_accuracy", "Overall CI Accuracy", "overall_accuracy"),
        ("skeleton_accuracy", "Skeleton CI Accuracy", "skeleton_accuracy"),
        ("overall_f1", "Overall F1", "overall_f1"),
    ]

    html += "<tr><th>Metric</th><th>Value</th><th>Rating</th></tr>"

    for key, label, rating_key in metrics:
        value = result.get(key)
        if value is not None:
            if rating_key:
                rating, _ = get_quality_rating(rating_key, value)
                rating_class = f"rating-{rating.lower()}"
                rating_html = f'<span class="{rating_class}">{rating}</span>'
            else:
                rating_html = "—"

            html += f"<tr>"
            html += f"<td>{label}</td>"
            html += f"<td>{value:.4f}</td>"
            html += f"<td>{rating_html}</td>"
            html += f"</tr>"

    # Add confusion matrix if available
    if "confusion_matrix" in result:
        cm = result["confusion_matrix"]
        html += "<tr><td colspan='3'><strong>Confusion Matrix:</strong> "
        html += f"TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']}</td></tr>"

    html += "</table>"
    return html


def _generate_umap_section(umap_paths, output_dir):
    """Generate HTML for UMAP visualizations."""
    html = '<div class="umap-grid">'

    for name, path in umap_paths.items():
        if path and Path(path).exists():
            rel_path = Path(path).name
            html += f"<div>"
            html += f"<h4>{name}</h4>"
            html += f'<img src="{rel_path}" alt="{name}">'
            html += f"</div>"

    html += "</div>"
    return html
