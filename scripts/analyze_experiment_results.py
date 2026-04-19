#!/usr/bin/env python3
"""
Analyze experiment results from eval_linear/ and eval_nonlinear/ directories.
Generate comprehensive HTML report with performance comparison across configs.
"""

import os
import re
from pathlib import Path
from datetime import datetime
import json
import base64
import numpy as np

# Config definitions from test_fedcdh_benchmark.py
CONFIG_DEFINITIONS = {
    "quick": {
        "d": 5,
        "K": 2,
        "n": 200,
        "epochs": 20,
        "description": "Quick smoke test: 5 vars, 2 clients, 200 samples",
    },
    "small": {
        "d": 8,
        "K": 3,
        "n": 600,
        "epochs": 50,
        "description": "Small-scale: 8 vars, 3 clients, 600 samples",
    },
    "medium": {
        "d": 10,
        "K": 3,
        "n": 1200,
        "epochs": 100,
        "description": "Medium-scale: 10 vars, 3 clients, 1200 samples",
    },
    "large": {
        "d": 11,
        "K": 5,
        "n": 1650,
        "epochs": 150,
        "description": "Large-scale: 11 vars, 5 clients, 1650 samples (Sachs-like: 330/client)",
    },
    "sachs": {
        "d": 11,
        "K": 3,
        "n": 853,
        "epochs": 100,
        "description": "Real Sachs dataset: 11 protein markers, 3 clients",
    },
}


def identify_config(d, K, n):
    """Identify which config this experiment matches."""
    for config_name, config_def in CONFIG_DEFINITIONS.items():
        if (
            config_def["d"] == d
            and config_def["K"] == K
            and abs(config_def["n"] - n) <= 50
        ):
            return config_name
    return "custom"


def encode_image_to_base64(image_path):
    """Encode image to base64 for embedding in HTML."""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        print(f"Warning: Failed to encode {image_path}: {e}")
        return None


def parse_run_log(log_path, exp_dir):
    """Parse a run.log file and extract all metrics + image paths."""
    with open(log_path, "r") as f:
        content = f.read()

    # Parse configuration
    config = {}
    config["scenario"] = re.search(r"Scenario: (\w+)", content).group(1)
    config["K"] = int(re.search(r"Clients \(K\): (\d+)", content).group(1))
    config["d"] = int(re.search(r"Features \(d\): (\d+)", content).group(1))
    config["n"] = int(re.search(r"Total samples: (\d+)", content).group(1))
    config["ci_method"] = re.search(r"CI method: (\w+)", content).group(1)
    config["alpha"] = float(re.search(r"Alpha: ([\d.]+)", content).group(1))
    config["epochs"] = int(re.search(r"SPN epochs: (\d+)", content).group(1))

    # Extract timestamp from directory name
    dir_name = Path(log_path).parent.name
    timestamp_str = dir_name.split("_")[0] + dir_name.split("_")[1]
    config["timestamp"] = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
    config["run_id"] = dir_name

    # Identify config size
    config["config_size"] = identify_config(config["d"], config["K"], config["n"])

    # Parse local SPN metrics
    local_spns = []
    for client_id in range(config["K"]):
        # Updated pattern to handle both formats:
        # [Local SPN Client 0] and [Local SPN Client 0 (Features [0, 1, 2])]
        # Use .*? to match any characters (including nested brackets) lazily
        pattern = rf"\[Local SPN Client {client_id}.*?\] Quality Metrics:.*?Train LL: ([-\d.]+).*?MMD p-value: ([\d.]+).*?KS test: (\d+)% failed"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            local_metrics = {
                "client_id": client_id,
                "train_ll": float(match.group(1)),
                "mmd_pvalue": float(match.group(2)),
                "ks_fail_ratio": float(match.group(3)) / 100.0,
            }

            # Parse independence structure for this client
            # Updated pattern to handle both formats
            indep_pattern = rf"\[Local SPN Client {client_id}.*?\] Independence Structure:.*?Tests: (\d+) total.*?Overall Accuracy: ([\d.]+).*?Overall F1: ([\d.]+).*?Skeleton Accuracy: ([\d.]+).*?Confusion: TP=(\d+), FP=(\d+), FN=(\d+), TN=(\d+)"
            indep_match = re.search(indep_pattern, content, re.DOTALL)
            if indep_match:
                local_metrics.update(
                    {
                        "total_tests": int(indep_match.group(1)),
                        "overall_accuracy": float(indep_match.group(2)),
                        "overall_f1": float(indep_match.group(3)),
                        "skeleton_accuracy": float(indep_match.group(4)),
                        "tp": int(indep_match.group(5)),
                        "fp": int(indep_match.group(6)),
                        "fn": int(indep_match.group(7)),
                        "tn": int(indep_match.group(8)),
                    }
                )

            # Get UMAP image for this client
            umap_path = exp_dir / f"umap_local_client_{client_id}.png"
            if umap_path.exists():
                local_metrics["umap_base64"] = encode_image_to_base64(umap_path)

            local_spns.append(local_metrics)

    # Parse global SPN metrics
    global_pattern = r"\[Global Federated SPN\] Quality Metrics:.*?Train LL: ([-\d.]+).*?MMD p-value: ([\d.]+).*?KS test: (\d+)% failed"
    global_match = re.search(global_pattern, content, re.DOTALL)

    global_spn = {}
    if global_match:
        global_spn = {
            "train_ll": float(global_match.group(1)),
            "mmd_pvalue": float(global_match.group(2)),
            "ks_fail_ratio": float(global_match.group(3)) / 100.0,
        }

        # Parse global independence structure
        global_indep_pattern = r"\[Global Federated SPN\] Independence Structure:.*?Tests: (\d+) total.*?Overall Accuracy: ([\d.]+).*?Overall F1: ([\d.]+).*?Skeleton Accuracy: ([\d.]+).*?Confusion: TP=(\d+), FP=(\d+), FN=(\d+), TN=(\d+)"
        global_indep_match = re.search(global_indep_pattern, content, re.DOTALL)
        if global_indep_match:
            global_spn.update(
                {
                    "total_tests": int(global_indep_match.group(1)),
                    "overall_accuracy": float(global_indep_match.group(2)),
                    "overall_f1": float(global_indep_match.group(3)),
                    "skeleton_accuracy": float(global_indep_match.group(4)),
                    "tp": int(global_indep_match.group(5)),
                    "fp": int(global_indep_match.group(6)),
                    "fn": int(global_indep_match.group(7)),
                    "tn": int(global_indep_match.group(8)),
                }
            )

        # Get global UMAP image
        global_umap_path = exp_dir / "umap_global_spn.png"
        if global_umap_path.exists():
            global_spn["umap_base64"] = encode_image_to_base64(global_umap_path)

    return {"config": config, "local_spns": local_spns, "global_spn": global_spn}


def collect_experiments(base_dir):
    """Collect all experiments from a base directory."""
    experiments = []
    base_path = Path(base_dir)

    if not base_path.exists():
        return experiments

    for exp_dir in sorted(base_path.iterdir()):
        if exp_dir.is_dir():
            log_path = exp_dir / "run.log"
            if log_path.exists():
                try:
                    exp_data = parse_run_log(log_path, exp_dir)
                    experiments.append(exp_data)
                except Exception as e:
                    print(f"Warning: Failed to parse {log_path}: {e}")

    return experiments


def get_metric_class(metric_name, value):
    """Get CSS class for metric based on thresholds."""
    thresholds = {
        "train_ll": {"good": -8.0, "fair": -12.0},
        "overall_f1": {"good": 0.6, "fair": 0.4},
        "skeleton_accuracy": {"good": 0.75, "fair": 0.6},
        "overall_accuracy": {"good": 0.75, "fair": 0.6},
        "mmd_pvalue": {"good": 0.05, "fair": 0.01},
        "ks_fail_ratio": {"good": 0.3, "fair": 0.5},
    }

    if metric_name not in thresholds:
        return ""

    thresh = thresholds[metric_name]

    if metric_name in ["ks_fail_ratio"]:
        if value <= thresh["good"]:
            return "metric-good"
        elif value <= thresh["fair"]:
            return "metric-fair"
        else:
            return "metric-poor"
    else:
        if value >= thresh["good"]:
            return "metric-good"
        elif value >= thresh["fair"]:
            return "metric-fair"
        else:
            return "metric-poor"


def compute_summary_stats(experiments):
    """Compute summary statistics for a set of experiments."""
    if not experiments:
        return {}

    stats = {}
    for config_size in ["small", "medium", "large"]:
        config_exps = [
            exp for exp in experiments if exp["config"]["config_size"] == config_size
        ]
        if not config_exps:
            continue

        # Global metrics
        global_metrics = {
            "train_ll": [],
            "overall_f1": [],
            "skeleton_accuracy": [],
            "overall_accuracy": [],
        }

        for exp in config_exps:
            global_spn = exp["global_spn"]
            for metric in global_metrics.keys():
                if metric in global_spn:
                    global_metrics[metric].append(global_spn[metric])

        # Calculate averages
        stats[config_size] = {}
        for metric, values in global_metrics.items():
            if values:
                stats[config_size][metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

    return stats


def generate_architecture_tab():
    """Generate SPN architecture reference tab showing all config×mode combinations."""

    # Calculate architectures for each config×mode
    configs = {
        "small": {"d": 8, "K": 3},
        "medium": {"d": 10, "K": 3},
        "large": {"d": 11, "K": 5},
    }

    # Calculate vertical feature splits
    vertical_splits = {
        "small": [2, 2, 4],  # d=8, K=3: [0-1], [2-3], [4-7]
        "medium": [3, 3, 4],  # d=10, K=3: [0-2], [3-5], [6-9]
        "large": [2, 2, 2, 2, 3],  # d=11, K=5: [0-1], [2-3], [4-5], [6-7], [8-10]
    }

    html = """
    <div id="main-architecture" class="main-tab-content">
        <div style="padding: 30px; max-width: 1400px; margin: 0 auto;">
            <h2>🏗️ SPN Architecture Guide (v1 Baseline)</h2>
            <p style="margin-bottom: 20px; color: #666;">
                This table shows the FIXED Einet (RAT-SPN) architecture used in v1 baseline experiments.
                All experiments used <code>num_sums=20, num_leaves=20</code> regardless of config or mode.
            </p>

            <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 4px solid #ffc107;">
                <h3 style="margin: 0 0 10px 0; color: #856404;">⚠️ Fixed Architecture (No Adaptive Scaling)</h3>
                <pre style="background: white; padding: 12px; border-radius: 4px; font-family: monospace; font-size: 13px; overflow-x: auto;">num_sums = 20           # FIXED for all experiments
num_leaves = 20         # FIXED for all experiments
num_repetitions = 10    # FIXED for all experiments
depth = max(1, floor(log2(local_d)))  # Only parameter that adapts

local_d = num_features_per_client + 1  # +1 for context column U</pre>
                <p style="margin: 10px 0 0 0; font-size: 13px; color: #856404;">
                    <strong>Result:</strong> This fixed architecture works for SMALL (d≤8, K≤3) but causes complete failure
                    on LARGE (d=11, K=5) with F1=0.000 across all modes. Adaptive scaling needed for v2.
                </p>
            </div>
"""

    for config_name, config_def in configs.items():
        d = config_def["d"]
        K = config_def["K"]

        # v1 baseline: FIXED architecture (no adaptive scaling)
        fixed_sums = 20
        fixed_leaves = 20

        # Horizontal
        local_d_horiz = d + 1
        depth_horiz = max(1, int(np.floor(np.log2(local_d_horiz))))

        # Vertical calculations (per client) - still fixed 20/20
        vert_clients = []
        for feat_count in vertical_splits[config_name]:
            local_d_vert = feat_count + 1
            depth_vert = max(1, int(np.floor(np.log2(local_d_vert))))
            vert_clients.append(
                {
                    "features": feat_count,
                    "local_d": local_d_vert,
                    "sums": fixed_sums,  # Fixed at 20
                    "leaves": fixed_leaves,  # Fixed at 20
                    "depth": depth_vert,
                }
            )

        html += f"""
            <div style="margin-bottom: 40px;">
                <h3 style="color: #667eea; margin-bottom: 15px;">
                    {config_name.upper()} Config (d={d}, K={K})
                </h3>

                <table style="width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: #667eea; color: white;">
                            <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Mode</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Client</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Features</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">local_d</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">scale_factor</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">num_sums</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">num_leaves</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">depth</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="background: #e8f0ff;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: 600;" rowspan="1">Horizontal</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">All clients</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{d}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;"><strong>{local_d_horiz}</strong></td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">N/A</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; background: #c3f0c3; font-weight: 600;">{fixed_sums}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; background: #c3f0c3; font-weight: 600;">{fixed_leaves}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{depth_horiz}</td>
                        </tr>
"""

        for i, client in enumerate(vert_clients):
            rowspan = len(vert_clients) if i == 0 else 0
            row_style = "background: #fff8e1;" if i % 2 == 0 else "background: #fffbf0;"

            if i == 0:
                html += f"""
                        <tr style="{row_style}">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: 600;" rowspan="{rowspan}">Vertical</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">Client {i}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{client['features']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;"><strong>{client['local_d']}</strong></td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">N/A</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; background: #ffebcc; font-weight: 600;">{client['sums']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; background: #ffebcc; font-weight: 600;">{client['leaves']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{client['depth']}</td>
                        </tr>
"""
            else:
                html += f"""
                        <tr style="{row_style}">
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">Client {i}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{client['features']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;"><strong>{client['local_d']}</strong></td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">N/A</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; background: #ffebcc; font-weight: 600;">{client['sums']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; background: #ffebcc; font-weight: 600;">{client['leaves']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{client['depth']}</td>
                        </tr>
"""

        html += f"""
                        <tr style="background: #f0e8ff;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: 600;">Hybrid</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;" colspan="7">
                                <em>Variable architecture per feature group (similar to vertical within each group)</em>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
"""

    html += """
            <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin-top: 30px;">
                <h4 style="margin: 0 0 10px 0; color: #856404;">💡 Key Observations (v1 Fixed Architecture)</h4>
                <ul style="margin: 5px 0; padding-left: 20px; color: #856404;">
                    <li><strong>All modes use identical architecture</strong>: num_sums=20, num_leaves=20 (no adaptive scaling)</li>
                    <li><strong>Only depth varies</strong>: Calculated as floor(log2(local_d)), ranging from 1-3</li>
                    <li><strong>Horizontal mode</strong>: Fixed 20/20 insufficient for LARGE (d=11, K=5) → F1=0.000</li>
                    <li><strong>Vertical mode</strong>: Fixed 20/20 over-parameterizes clients with 2-3 features → overfitting on linear data</li>
                    <li><strong>Result</strong>: Fixed architecture only works for SMALL (d≤8, K≤3). LARGE completely fails.</li>
                    <li><strong>Implication</strong>: v2 must implement adaptive scaling based on dimensionality, mode, and sample size</li>
                </ul>
            </div>
        </div>
    </div>
"""

    return html


def generate_html_report(linear_exps, nonlinear_exps, output_path):
    """Generate comprehensive HTML report with nested tabs."""

    html = (
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FedCDH Experiment Analysis Report</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8fafc;
            color: #1e293b;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        .header h1 {
            margin: 0 0 15px 0;
            font-size: 36px;
            font-weight: 700;
        }
        .header p {
            margin: 5px 0;
            opacity: 0.95;
            font-size: 16px;
        }

        /* Config reference box */
        .config-ref {
            background: white;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .config-ref h3 {
            margin: 0 0 15px 0;
            color: #667eea;
            font-size: 18px;
        }
        .config-ref-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .config-ref-item {
            padding: 12px;
            background: #f8fafc;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        }
        .config-ref-item strong {
            color: #667eea;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .config-ref-item p {
            margin: 5px 0 0 0;
            font-size: 13px;
            color: #64748b;
        }

        /* Main tabs */
        .main-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 0;
        }
        .main-tab {
            padding: 16px 32px;
            background: white;
            border: none;
            border-radius: 12px 12px 0 0;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.05);
            color: #64748b;
        }
        .main-tab:hover {
            background: #f1f5f9;
            transform: translateY(-2px);
        }
        .main-tab.active {
            background: #667eea;
            color: white;
            box-shadow: 0 -4px 12px rgba(102, 126, 234, 0.3);
        }
        .main-tab-content {
            display: none;
            background: white;
            padding: 0;
            border-radius: 0 12px 12px 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .main-tab-content.active {
            display: block;
        }

        /* Sub-tabs */
        .sub-tabs {
            display: flex;
            gap: 5px;
            padding: 20px 20px 0 20px;
            background: #f8fafc;
            border-bottom: 2px solid #e2e8f0;
        }
        .sub-tab {
            padding: 12px 24px;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
            color: #64748b;
        }
        .sub-tab:hover {
            background: #f1f5f9;
        }
        .sub-tab.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .sub-tab-content {
            display: none;
            padding: 30px;
        }
        .sub-tab-content.active {
            display: block;
        }

        /* Config badges */
        .config-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .config-small { background: #dbeafe; color: #1e40af; }
        .config-medium { background: #fef3c7; color: #92400e; }
        .config-large { background: #fee2e2; color: #991b1b; }

        /* Experiment cards */
        .experiment-card {
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            margin-bottom: 25px;
            background: white;
            overflow: hidden;
        }
        .experiment-header {
            padding: 20px 25px;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #fafbfc;
            border-bottom: 1px solid #e2e8f0;
        }
        .experiment-header:hover {
            background: #f1f5f9;
        }
        .experiment-title {
            font-size: 16px;
            font-weight: 600;
            color: #1e293b;
        }
        .experiment-meta {
            font-size: 13px;
            color: #64748b;
            margin-top: 5px;
        }
        .expand-icon {
            font-size: 20px;
            color: #94a3b8;
            transition: transform 0.3s;
        }
        .experiment-card.expanded .expand-icon {
            transform: rotate(180deg);
        }

        .scenario-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 8px;
            text-transform: uppercase;
        }
        .scenario-horizontal { background: #dbeafe; color: #1e40af; }
        .scenario-vertical { background: #fce7f3; color: #9f1239; }
        .scenario-hybrid { background: #fef3c7; color: #92400e; }

        /* Performance section */
        .performance-section {
            padding: 25px;
        }
        .performance-section h3 {
            margin: 0 0 20px 0;
            color: #667eea;
            font-size: 18px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        .metric-card {
            background: #fafbfc;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        .metric-label {
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 22px;
            font-weight: 700;
        }
        .metric-good { color: #10b981; }
        .metric-fair { color: #f59e0b; }
        .metric-poor { color: #ef4444; }

        /* UMAP - Horizontal grid layout */
        .umap-section {
            margin: 25px 0;
        }
        .umap-section h4 {
            margin: 0 0 15px 0;
            color: #475569;
            font-size: 16px;
        }
        .umap-vertical {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        .umap-item {
            text-align: center;
            padding: 15px;
            background: #fafbfc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        .umap-item img {
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .umap-label {
            margin-top: 10px;
            font-size: 14px;
            font-weight: 600;
            color: #1e293b;
            background: white;
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        }

        /* Local SPNs section */
        .local-spns {
            display: none;
            padding: 25px;
            background: #f8fafc;
            border-top: 2px solid #e2e8f0;
        }
        .experiment-card.expanded .local-spns {
            display: block;
        }
        .local-spns h3 {
            margin: 0 0 20px 0;
            color: #475569;
            font-size: 18px;
        }
        .local-client-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        .local-client-card h4 {
            margin: 0 0 15px 0;
            color: #1e293b;
            font-size: 16px;
        }

        .confusion-matrix {
            display: inline-block;
            margin-top: 10px;
            padding: 10px;
            background: #f8fafc;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }

        /* Summary comparison table */
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .summary-table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        .summary-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }
        .summary-table tr:hover {
            background: #f9f9f9;
        }
    </style>
    <script>
        function showMainTab(tabName) {
            document.querySelectorAll('.main-tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.querySelectorAll('.main-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById('main-' + tabName).classList.add('active');
            document.querySelector(`[onclick="showMainTab('${tabName}')"]`).classList.add('active');
        }

        function showSubTab(mainTab, subTab) {
            const prefix = mainTab + '-';
            document.querySelectorAll(`[id^="${prefix}sub-"]`).forEach(content => {
                content.classList.remove('active');
            });
            document.querySelectorAll(`[data-main="${mainTab}"].sub-tab`).forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById(prefix + 'sub-' + subTab).classList.add('active');
            document.querySelector(`[onclick="showSubTab('${mainTab}', '${subTab}')"]`).classList.add('active');
        }

        function toggleExperiment(id) {
            const card = document.getElementById(id);
            card.classList.toggle('expanded');
        }
    </script>
</head>
<body>
    <div class="header">
        <h1>🔬 FedCDH Experiment Analysis Report</h1>
        <p><strong>Generated:</strong> """
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + """</p>
        <p><strong>Branch:</strong> fedpc</p>
        <p><strong>Total Experiments:</strong> """
        + str(len(linear_exps) + len(nonlinear_exps))
        + """ ("""
        + str(len(linear_exps))
        + """ linear + """
        + str(len(nonlinear_exps))
        + """ nonlinear)</p>
    </div>

    <div class="config-ref">
        <h3>📋 Configuration Reference</h3>
        <div class="config-ref-grid">
"""
    )

    for config_name in ["small", "medium", "large"]:
        config_def = CONFIG_DEFINITIONS[config_name]
        d = config_def["d"]

        # v1 baseline: FIXED hyperparameters (no adaptive scaling)
        html += f"""
            <div class="config-ref-item">
                <strong>{config_name.upper()}</strong>
                <p>{config_def['description']}</p>
                <p style="margin-top: 8px; font-family: monospace; font-size: 12px;">
                    d={config_def['d']}, K={config_def['K']}, n={config_def['n']}, epochs={config_def['epochs']}
                </p>
                <p style="margin-top: 8px; font-size: 11px; color: #667eea; font-weight: 600;">
                    SPN: num_sums=20, num_leaves=20, depth=calculated, num_reps=10
                </p>
                <p style="margin-top: 2px; font-size: 10px; color: #888;">
                    ⓘ FIXED architecture (no adaptive scaling in v1)
                </p>
            </div>
"""

    html += (
        """
        </div>
    </div>

    <div class="main-tabs">
        <button class="main-tab active" onclick="showMainTab('linear')">📊 Linear Data ("""
        + str(len(linear_exps))
        + """ experiments)</button>
        <button class="main-tab" onclick="showMainTab('nonlinear')">📈 Nonlinear Data ("""
        + str(len(nonlinear_exps))
        + """ experiments)</button>
        <button class="main-tab" onclick="showMainTab('architecture')">🏗️ SPN Architecture</button>
    </div>
"""
    )

    # Generate linear tab
    html += generate_main_tab_content("linear", linear_exps)

    # Generate nonlinear tab
    html += generate_main_tab_content("nonlinear", nonlinear_exps)

    # Generate architecture reference tab
    html += generate_architecture_tab()

    html += """
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"✅ HTML report generated: {output_path}")


def generate_main_tab_content(tab_id, experiments):
    """Generate content for one main tab with nested sub-tabs."""

    if not experiments:
        return f"""
    <div id="main-{tab_id}" class="main-tab-content {'active' if tab_id == 'linear' else ''}">
        <p style="padding: 30px;">No experiments found.</p>
    </div>
"""

    # Group experiments by config size
    grouped = {}
    for exp in experiments:
        config_size = exp["config"]["config_size"]
        if config_size not in grouped:
            grouped[config_size] = []
        grouped[config_size].append(exp)

    html = f"""
    <div id="main-{tab_id}" class="main-tab-content {'active' if tab_id == 'linear' else ''}">
        <div class="sub-tabs">
"""

    # Generate sub-tab buttons
    for config_size in ["small", "medium", "large"]:
        count = len(grouped.get(config_size, []))
        is_active = "active" if config_size == "small" else ""
        html += f"""
            <button class="sub-tab {is_active}" data-main="{tab_id}" onclick="showSubTab('{tab_id}', '{config_size}')">
                <span class="config-badge config-{config_size}">{config_size.upper()}</span> ({count})
            </button>
"""

    html += f"""
            <button class="sub-tab" data-main="{tab_id}" onclick="showSubTab('{tab_id}', 'summary')">
                📊 SUMMARY
            </button>
        </div>
"""

    # Generate sub-tab contents
    for config_size in ["small", "medium", "large"]:
        is_active = "active" if config_size == "small" else ""
        html += generate_sub_tab_content(
            tab_id, config_size, grouped.get(config_size, []), is_active
        )

    # Generate summary sub-tab
    html += generate_summary_sub_tab(tab_id, grouped)

    html += """
    </div>
"""

    return html


def generate_grouped_bar_chart(
    mode_performance, metric_keys, metric_labels, title, normalize_eval=False
):
    """Generate a grouped bar chart SVG for metrics comparison across modes."""

    # Chart dimensions
    chart_width = 400
    chart_height = 250
    margin_left = 60
    margin_right = 20
    margin_top = 40
    margin_bottom = 60
    plot_width = chart_width - margin_left - margin_right
    plot_height = chart_height - margin_top - margin_bottom

    # Mode colors
    mode_colors = {"horizontal": "#3b82f6", "vertical": "#10b981", "hybrid": "#f59e0b"}

    # Prepare data
    modes = list(mode_performance.keys())
    num_metrics = len(metric_keys)
    num_modes = len(modes)
    group_width = plot_width / num_metrics
    bar_width = group_width / (num_modes + 1)

    svg = f"""
                        <div style="background: white; border-radius: 8px; padding: 15px;">
                            <h4 style="margin: 0 0 10px 0; color: #1e293b; font-size: 14px;">{title}</h4>
                            <svg width="{chart_width}" height="{chart_height}" style="font-family: sans-serif;">
"""

    # Draw axes
    svg += f"""
                                <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="2"/>
                                <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="2"/>
"""

    # Determine y-axis scale
    max_val = 1.0
    min_val = 0.0
    is_train_ll_chart = "train_ll" in metric_keys and len(metric_keys) == 1

    # For train LL chart, use actual negative values on y-axis
    if is_train_ll_chart:
        all_lls = [mode_performance[m].get("train_ll", 0.0) for m in modes]
        min_ll = min(all_lls) if all_lls else -20
        max_ll = max(all_lls) if all_lls else 0
        # Add some padding
        range_padding = (max_ll - min_ll) * 0.1
        min_val = min_ll - range_padding
        max_val = max_ll + range_padding
        # Ensure max goes to 0 or slightly above if all values are negative
        if max_val < 0:
            max_val = 0

    # Draw y-axis labels
    for i in range(6):
        y_pos = margin_top + plot_height - (i * plot_height / 5)
        if is_train_ll_chart:
            # Show actual negative values for Train LL
            val = min_val + (i * (max_val - min_val) / 5)
            svg += f"""
                                <text x="{margin_left - 10}" y="{y_pos + 4}" text-anchor="end" fill="#1e293b" font-size="10">{val:.1f}</text>
                                <line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + plot_width}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="2,2"/>
"""
        else:
            # Show 0-1 scale for other metrics
            val = i * 0.2
            svg += f"""
                                <text x="{margin_left - 10}" y="{y_pos + 4}" text-anchor="end" fill="#1e293b" font-size="10">{val:.1f}</text>
                                <line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + plot_width}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="2,2"/>
"""

    # Draw bars
    for metric_idx, (metric_key, metric_label) in enumerate(
        zip(metric_keys, metric_labels)
    ):
        group_x = margin_left + metric_idx * group_width

        for mode_idx, mode in enumerate(modes):
            value = mode_performance[mode].get(metric_key, 0.0)

            # Calculate bar position and height
            if is_train_ll_chart:
                # For Train LL chart with negative values, bars extend downward from 0
                # Calculate where this value appears on the y-axis
                if max_val != min_val:
                    # Position of the value on the scale (0.0 is at top, min_val is at bottom)
                    value_y_pos = margin_top + plot_height * (max_val - value) / (
                        max_val - min_val
                    )
                else:
                    value_y_pos = margin_top + plot_height / 2

                # Bar starts at 0 line (top) and extends down to value position
                bar_y = margin_top  # Start at 0.0 line
                bar_height = (
                    value_y_pos - margin_top
                )  # Extend downward to value position
                bar_x = group_x + mode_idx * bar_width + bar_width * 0.5
            else:
                # For positive metrics (0-1 scale), bars extend upward from bottom
                if metric_key == "ks_fail_ratio":
                    norm_value = value / 100.0  # Convert percentage to [0, 1]
                elif metric_key in [
                    "mmd_pval",
                    "precision",
                    "recall",
                    "f1",
                    "skel_acc",
                    "overall_acc",
                ]:
                    norm_value = value  # Already [0, 1]
                else:
                    norm_value = value

                bar_height = norm_value * plot_height
                bar_x = group_x + mode_idx * bar_width + bar_width * 0.5
                bar_y = margin_top + plot_height - bar_height

            # Format value based on metric type - use original precision
            if metric_key == "ks_fail_ratio":
                value_text = f"{value:.0f}%"
            elif metric_key == "train_ll":
                value_text = f"{value:.4f}"  # Show 4 decimals for Train LL
            else:
                value_text = f"{value:.3f}"  # Show 3 decimals for other metrics

            # Position label - above bar for positive metrics, below bar for negative (Train LL)
            if is_train_ll_chart:
                label_y = bar_y + bar_height + 12  # Below the bar
            else:
                label_y = bar_y - 3  # Above the bar

            svg += f"""
                                <rect x="{bar_x}" y="{bar_y}" width="{bar_width * 0.8}" height="{bar_height}" fill="{mode_colors[mode]}" opacity="0.8" rx="2"/>
                                <text x="{bar_x + bar_width * 0.4}" y="{label_y}" text-anchor="middle" fill="#1e293b" font-size="9">{value_text}</text>
"""

        # Draw metric label
        label_x = group_x + group_width / 2
        svg += f"""
                                <text x="{label_x}" y="{margin_top + plot_height + 20}" text-anchor="middle" fill="#1e293b" font-size="11">{metric_label}</text>
"""

    # Draw legend
    legend_y = margin_top + plot_height + 40
    legend_x_start = margin_left
    for idx, mode in enumerate(modes):
        legend_x = legend_x_start + idx * 100
        svg += f"""
                                <rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{mode_colors[mode]}" opacity="0.8" rx="2"/>
                                <text x="{legend_x + 16}" y="{legend_y + 10}" fill="#1e293b" font-size="10">{mode.upper()}</text>
"""

    svg += """
                            </svg>
                        </div>
"""

    return svg


def generate_sub_tab_content(main_tab, config_size, experiments, is_active):
    """Generate content for one sub-tab (one config size)."""

    html = f"""
        <div id="{main_tab}-sub-{config_size}" class="sub-tab-content {is_active}">
"""

    if not experiments:
        html += "<p>No experiments found for this configuration.</p>"
    else:
        # Calculate performance summary for this config
        # Extract GLOBAL SPN metrics from each mode (horizontal/vertical/hybrid)
        mode_performance = {}
        for exp in experiments:
            scenario = exp["config"]["scenario"]  # Mode: horizontal/vertical/hybrid
            global_spn = exp["global_spn"]  # GLOBAL Federated SPN metrics only

            # Calculate precision and recall from GLOBAL SPN confusion matrix
            tp = global_spn.get("tp", 0)
            fp = global_spn.get("fp", 0)
            fn = global_spn.get("fn", 0)
            tn = global_spn.get("tn", 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Collect all metrics from GLOBAL SPN only (not local SPNs)
            if scenario not in mode_performance:
                mode_performance[scenario] = {
                    # Causal discovery metrics from GLOBAL SPN
                    "f1": global_spn.get("overall_f1", 0.0),
                    "skel_acc": global_spn.get(
                        "skeleton_accuracy", 0.0
                    ),  # Fixed: was 'skeleton_acc'
                    "overall_acc": global_spn.get(
                        "overall_accuracy", 0.0
                    ),  # Fixed: was 'overall_acc'
                    # Evaluation metrics from GLOBAL SPN
                    "train_ll": global_spn.get("train_ll", 0.0),
                    "mmd_pval": global_spn.get(
                        "mmd_pvalue", 0.0
                    ),  # Fixed: was 'mmd_pval'
                    "ks_fail_ratio": global_spn.get("ks_fail_ratio", 0.0),
                    "precision": precision,
                    "recall": recall,
                }

        # Find best mode by F1 score
        if mode_performance:
            best_mode = max(mode_performance.items(), key=lambda x: x[1]["f1"])
            worst_mode = min(mode_performance.items(), key=lambda x: x[1]["f1"])

            # Determine status
            if best_mode[1]["f1"] >= 0.5:
                status_color = "#10b981"
                status_icon = "✅"
                status_text = "Good Performance"
            elif best_mode[1]["f1"] >= 0.3:
                status_color = "#f59e0b"
                status_icon = "⚠️"
                status_text = "Moderate Performance"
            elif best_mode[1]["f1"] > 0.0:
                status_color = "#ef4444"
                status_icon = "❌"
                status_text = "Poor Performance"
            else:
                status_color = "#dc2626"
                status_icon = "🔴"
                status_text = "Complete Failure"

            # Add performance summary box
            html += f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 25px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 15px 0; font-size: 18px; display: flex; align-items: center; gap: 10px;">
                    {status_icon} {config_size.upper()} Config Performance Summary
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 8px;">
                        <div style="font-size: 12px; opacity: 0.9; margin-bottom: 5px;">🏆 Best Mode</div>
                        <div style="font-size: 20px; font-weight: 600;">{best_mode[0].upper()}</div>
                        <div style="font-size: 14px; margin-top: 5px;">F1: {best_mode[1]['f1']:.3f}</div>
                    </div>
"""

            # Add comparison with other modes
            for mode, metrics in sorted(
                mode_performance.items(), key=lambda x: x[1]["f1"], reverse=True
            ):
                if mode != best_mode[0]:
                    html += f"""
                    <div style="background: rgba(255,255,255,0.10); padding: 12px; border-radius: 8px;">
                        <div style="font-size: 12px; opacity: 0.9; margin-bottom: 5px;">{mode.upper()}</div>
                        <div style="font-size: 16px;">F1: {metrics['f1']:.3f}</div>
                        <div style="font-size: 12px; opacity: 0.8;">Skel: {metrics['skel_acc']:.3f}, Overall: {metrics['overall_acc']:.3f}</div>
                    </div>
"""

            html += f"""
                </div>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2); font-size: 13px;">
                    <strong style="color: {status_color};">{status_text}</strong> -
                    Best F1: {best_mode[1]['f1']:.3f}, Worst F1: {worst_mode[1]['f1']:.3f},
                    Range: {best_mode[1]['f1'] - worst_mode[1]['f1']:.3f}
                </div>

                <!-- Performance Charts -->
                <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.3);">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
"""

            # Generate Train LL Chart (left)
            html += generate_grouped_bar_chart(
                mode_performance,
                ["train_ll"],
                ["Train LL"],
                "📊 Train Log-Likelihood",
                normalize_eval=True,
            )

            # Generate Global SPN Quality Metrics Chart (middle)
            # Shows: MMD p-value, KS Fail %, Precision, Recall from GLOBAL SPN per mode
            html += generate_grouped_bar_chart(
                mode_performance,
                ["mmd_pval", "ks_fail_ratio", "precision", "recall"],
                ["MMD p-val", "KS Fail %", "Precision", "Recall"],
                "📈 Global SPN Quality Metrics",
                normalize_eval=True,
            )

            # Generate Causal Discovery Metrics Chart (right)
            # Shows: Overall F1, Skeleton Acc, Overall Acc from GLOBAL SPN per mode
            html += generate_grouped_bar_chart(
                mode_performance,
                ["f1", "skel_acc", "overall_acc"],
                ["Overall F1", "Skeleton Acc", "Overall Acc"],
                "🎯 Causal Discovery Metrics",
                normalize_eval=False,
            )

            html += """
                    </div>
                </div>
            </div>
"""
        # Generate experiment cards
        # Generate experiment cards
        for i, exp in enumerate(experiments):
            exp_id = f"{main_tab}_{config_size}_{i}"
            config = exp["config"]
            global_spn = exp["global_spn"]

            html += f"""
            <div class="experiment-card" id="{exp_id}">
                <div class="experiment-header" onclick="toggleExperiment('{exp_id}')">
                    <div>
                        <div class="experiment-title">
                            <span class="scenario-badge scenario-{config['scenario']}">{config['scenario']}</span>
                            {config['run_id']}
                        </div>
                        <div class="experiment-meta">
                            Run: {config['timestamp'].strftime('%Y-%m-%d %H:%M')} |
                            K={config['K']} clients, d={config['d']} vars, n={config['n']} samples
                        </div>
                    </div>
                    <div class="expand-icon">▼</div>
                </div>

                <div class="performance-section">
                    <h3>🌐 Global Federated SPN Performance</h3>

                    <div class="metrics-grid">
"""

            # Add metric cards
            metrics = [
                ("Train LL", "train_ll", global_spn.get("train_ll")),
                ("Overall F1", "overall_f1", global_spn.get("overall_f1")),
                (
                    "Skeleton Acc",
                    "skeleton_accuracy",
                    global_spn.get("skeleton_accuracy"),
                ),
                ("Overall Acc", "overall_accuracy", global_spn.get("overall_accuracy")),
                ("MMD p-value", "mmd_pvalue", global_spn.get("mmd_pvalue")),
                (
                    "KS Fail %",
                    "ks_fail_ratio",
                    global_spn.get("ks_fail_ratio", 0) * 100,
                ),
            ]

            for label, metric_key, value in metrics:
                if value is not None:
                    metric_class = get_metric_class(
                        metric_key,
                        value if metric_key != "ks_fail_ratio" else value / 100,
                    )
                    if metric_key == "ks_fail_ratio":
                        display_value = f"{value:.0f}%"
                    elif metric_key in ["train_ll"]:
                        display_value = f"{value:.2f}"
                    else:
                        display_value = f"{value:.3f}"

                    html += f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value {metric_class}">{display_value}</div>
                        </div>
"""

            if "tp" in global_spn:
                html += f"""
                        <div class="metric-card">
                            <div class="metric-label">Confusion Matrix</div>
                            <div class="confusion-matrix">
                                TP={global_spn['tp']}, FP={global_spn['fp']}<br>
                                FN={global_spn['fn']}, TN={global_spn['tn']}
                            </div>
                        </div>
"""

            html += """
                    </div>

                    <div class="umap-section">
                        <h4>📊 UMAP Visualizations</h4>
                        <div class="umap-vertical">
"""

            # Add global UMAP first
            if global_spn.get("umap_base64"):
                html += f"""
                            <div class="umap-item">
                                <img src="data:image/png;base64,{global_spn['umap_base64']}" alt="Global SPN UMAP">
                                <div class="umap-label">🌐 Global SPN</div>
                            </div>
"""

            # Add local UMAPs vertically
            for local_spn in exp["local_spns"]:
                if local_spn.get("umap_base64"):
                    html += f"""
                            <div class="umap-item">
                                <img src="data:image/png;base64,{local_spn['umap_base64']}" alt="Local Client {local_spn['client_id']} UMAP">
                                <div class="umap-label">🖥️ Local Client {local_spn['client_id']}</div>
                            </div>
"""

            html += """
                        </div>
                    </div>
                </div>

                <div class="local-spns">
                    <h3>🖥️ Local SPN Details</h3>
"""

            # Add local client cards
            for local_spn in exp["local_spns"]:
                html += f"""
                    <div class="local-client-card">
                        <h4>Client {local_spn['client_id']}</h4>
                        <div class="metrics-grid">
"""

                local_metrics = [
                    ("Train LL", "train_ll", local_spn.get("train_ll")),
                    ("Overall F1", "overall_f1", local_spn.get("overall_f1")),
                    (
                        "Skeleton Acc",
                        "skeleton_accuracy",
                        local_spn.get("skeleton_accuracy"),
                    ),
                    (
                        "Overall Acc",
                        "overall_accuracy",
                        local_spn.get("overall_accuracy"),
                    ),
                    ("MMD p-value", "mmd_pvalue", local_spn.get("mmd_pvalue")),
                    (
                        "KS Fail %",
                        "ks_fail_ratio",
                        local_spn.get("ks_fail_ratio", 0) * 100,
                    ),
                ]

                for label, metric_key, value in local_metrics:
                    if value is not None:
                        metric_class = get_metric_class(
                            metric_key,
                            value if metric_key != "ks_fail_ratio" else value / 100,
                        )
                        if metric_key == "ks_fail_ratio":
                            display_value = f"{value:.0f}%"
                        elif metric_key in ["train_ll"]:
                            display_value = f"{value:.2f}"
                        else:
                            display_value = f"{value:.3f}"

                        html += f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value {metric_class}">{display_value}</div>
                            </div>
"""

                if "tp" in local_spn:
                    html += f"""
                            <div class="metric-card">
                                <div class="metric-label">Confusion Matrix</div>
                                <div class="confusion-matrix">
                                    TP={local_spn['tp']}, FP={local_spn['fp']}<br>
                                    FN={local_spn['fn']}, TN={local_spn['tn']}
                                </div>
                            </div>
"""

                html += """
                        </div>
                    </div>
"""

            html += """
                </div>
            </div>
"""

    html += """
        </div>
"""

    return html


def generate_summary_sub_tab(main_tab, grouped):
    """Generate summary comparison tab."""

    stats = compute_summary_stats([exp for exps in grouped.values() for exp in exps])

    html = f"""
        <div id="{main_tab}-sub-summary" class="sub-tab-content">
            <h2 style="margin-top: 0; color: #1e293b;">📊 Performance Summary Across Configurations</h2>

            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Config</th>
                        <th>Experiments</th>
                        <th>Avg Train LL</th>
                        <th>Avg Overall F1</th>
                        <th>Avg Skeleton Acc</th>
                        <th>Avg Overall Acc</th>
                    </tr>
                </thead>
                <tbody>
"""

    for config_size in ["small", "medium", "large"]:
        if config_size not in stats:
            continue

        config_stats = stats[config_size]
        count = config_stats["train_ll"]["count"] if "train_ll" in config_stats else 0

        html += f"""
                    <tr>
                        <td><span class="config-badge config-{config_size}">{config_size.upper()}</span></td>
                        <td>{count}</td>
"""

        for metric in [
            "train_ll",
            "overall_f1",
            "skeleton_accuracy",
            "overall_accuracy",
        ]:
            if metric in config_stats:
                mean_val = config_stats[metric]["mean"]
                min_val = config_stats[metric]["min"]
                max_val = config_stats[metric]["max"]
                metric_class = get_metric_class(metric, mean_val)

                if metric == "train_ll":
                    html += f'<td class="{metric_class}">{mean_val:.2f} <span style="font-size: 11px; color: #94a3b8;">[{min_val:.2f}, {max_val:.2f}]</span></td>'
                else:
                    html += f'<td class="{metric_class}">{mean_val:.3f} <span style="font-size: 11px; color: #94a3b8;">[{min_val:.3f}, {max_val:.3f}]</span></td>'
            else:
                html += "<td>N/A</td>"

        html += """
                    </tr>
"""

    html += """
                </tbody>
            </table>

            <div style="margin-top: 30px; padding: 20px; background: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 8px;">
                <h3 style="margin: 0 0 10px 0; color: #92400e;">💡 Key Insights</h3>
                <ul style="margin: 5px 0; padding-left: 20px; color: #78350f;">
                    <li>Performance trends across config sizes (Small → Medium → Large)</li>
                    <li>Range [min, max] shows variability across different scenarios</li>
                    <li>Color coding: 🟢 Green = Good, 🟡 Yellow = Fair, 🔴 Red = Poor</li>
                </ul>
            </div>
"""

    # Add overall performance summary by mode
    mode_summary = {}
    for config_size, exps in grouped.items():
        for exp in exps:
            scenario = exp["config"]["scenario"]
            if scenario not in mode_summary:
                mode_summary[scenario] = {"configs": {}}
            if config_size not in mode_summary[scenario]["configs"]:
                mode_summary[scenario]["configs"][config_size] = []
            mode_summary[scenario]["configs"][config_size].append(
                exp["global_spn"].get("overall_f1", 0.0)
            )

    html += """
            <div style="margin-top: 30px;">
                <h2 style="color: #1e293b;">🏆 Mode Performance by Configuration</h2>
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Mode</th>
                            <th>SMALL Config</th>
                            <th>MEDIUM Config</th>
                            <th>LARGE Config</th>
                            <th>Overall Assessment</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for mode in ["horizontal", "vertical", "hybrid"]:
        if mode in mode_summary:
            html += f"""
                        <tr>
                            <td><span class="scenario-badge scenario-{mode}">{mode.upper()}</span></td>
"""
            # For each config size
            for config in ["small", "medium", "large"]:
                if config in mode_summary[mode]["configs"]:
                    f1_scores = mode_summary[mode]["configs"][config]
                    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                    metric_class = get_metric_class("overall_f1", avg_f1)
                    html += f'<td class="{metric_class}">F1: {avg_f1:.3f}</td>'
                else:
                    html += "<td>N/A</td>"

            # Overall assessment
            all_f1s = [
                f1
                for config_f1s in mode_summary[mode]["configs"].values()
                for f1 in config_f1s
            ]
            avg_overall = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0

            if avg_overall >= 0.5:
                assessment = "✅ Good"
                color = "#10b981"
            elif avg_overall >= 0.3:
                assessment = "⚠️ Moderate"
                color = "#f59e0b"
            elif avg_overall > 0.0:
                assessment = "❌ Poor"
                color = "#ef4444"
            else:
                assessment = "🔴 Failure"
                color = "#dc2626"

            html += f'<td style="color: {color}; font-weight: 600;">{assessment} (Avg: {avg_overall:.3f})</td>'
            html += """
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>

            <div style="margin-top: 30px; padding: 20px; background: #fee2e2; border-left: 4px solid #dc2626; border-radius: 8px;">
                <h3 style="margin: 0 0 15px 0; color: #991b1b;">🔍 Critical Findings (v1 Fixed Architecture)</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; color: #7f1d1d;">
                    <div style="background: white; padding: 15px; border-radius: 6px;">
                        <strong style="color: #dc2626;">SMALL Config (d=8, K=3)</strong>
                        <p style="margin: 8px 0 0 0; font-size: 14px;">Fixed 20/20 architecture WORKS. All modes functional (F1: 0.4-0.6 range).</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 6px;">
                        <strong style="color: #f59e0b;">MEDIUM Config (d=10, K=3)</strong>
                        <p style="margin: 8px 0 0 0; font-size: 14px;">Fixed 20/20 architecture STRUGGLES. Performance degrades significantly (F1: 0.1-0.4).</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 6px;">
                        <strong style="color: #7f1d1d;">LARGE Config (d=11, K=5)</strong>
                        <p style="margin: 8px 0 0 0; font-size: 14px;">Fixed 20/20 architecture FAILS COMPLETELY. All modes F1=0.000 (catastrophic failure).</p>
                    </div>
                </div>
                <p style="margin: 15px 0 0 0; font-weight: 600; color: #991b1b;">
                    ⚠️ Conclusion: Fixed architecture hits complexity ceiling at d×K ≈ 30-40. Adaptive scaling required for v2.
                </p>
            </div>
        </div>
"""

    return html


def main():
    # Define base directories
    project_root = Path(__file__).parent.parent

    # Check for v1 baseline experiments first
    v1_linear_dir = project_root / "experiments" / "v1_baseline_fixed20" / "eval_linear"
    v1_nonlinear_dir = (
        project_root / "experiments" / "v1_baseline_fixed20" / "eval_nonlinear"
    )

    if v1_linear_dir.exists() and v1_nonlinear_dir.exists():
        linear_dir = v1_linear_dir
        nonlinear_dir = v1_nonlinear_dir
        output_path = (
            project_root
            / "experiments"
            / "v1_baseline_fixed20"
            / "experiment_analysis_report.html"
        )
        print(
            "🔍 Analyzing v1 baseline experiments (fixed num_sums=20, num_leaves=20)..."
        )
    else:
        linear_dir = project_root / "eval_linear"
        nonlinear_dir = project_root / "eval_nonlinear"
        output_path = project_root / "experiment_analysis_report.html"
        print("🔍 Collecting experiments...")

    print(f"   Linear: {linear_dir}")
    print(f"   Nonlinear: {nonlinear_dir}")

    # Collect experiments
    linear_exps = collect_experiments(linear_dir)
    nonlinear_exps = collect_experiments(nonlinear_dir)

    print(f"   Found {len(linear_exps)} linear experiments")
    print(f"   Found {len(nonlinear_exps)} nonlinear experiments")

    if not linear_exps and not nonlinear_exps:
        print("❌ No experiments found!")
        return

    # Generate report
    print(f"\n📝 Generating HTML report...")
    generate_html_report(linear_exps, nonlinear_exps, output_path)

    print(f"\n✅ Report generated successfully!")
    print(f"   Open: {output_path}")


if __name__ == "__main__":
    main()
