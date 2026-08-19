from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score, roc_auc_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

try:  # pragma: no cover - optional visuals dependency
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:  # pragma: no cover - optional visuals dependency
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

HUMAN_FRIENDLY_LABELS = {
    "loading_unloading_time": "loading & unloading time",
    "customs_clearance_time": "customs clearance time",
    "order_fulfillment_status": "order fulfilment readiness",
    "historical_demand": "historical demand forecast",
    "supplier_reliability_score": "supplier reliability score",
    "port_congestion_level": "port congestion",
    "eta_variation_hours": "ETA variation (hrs)",
    "traffic_congestion_level": "traffic congestion",
    "driver_behavior_score": "driver behaviour score",
    "fatigue_monitoring_score": "fatigue monitoring score",
    "handling_equipment_availability": "equipment availability",
    "warehouse_inventory_level": "warehouse inventory level",
}


def pretty_feature(name: str) -> str:
    if name in HUMAN_FRIENDLY_LABELS:
        return HUMAN_FRIENDLY_LABELS[name]
    return name.replace("_", " ").title()


@dataclass
class CostModelOutputs:
    model: RandomForestRegressor
    imputer: SimpleImputer
    features: List[str]
    importances: pd.DataFrame
    metrics: Dict[str, float]


@dataclass
class DelayModelOutputs:
    model: RandomForestClassifier
    imputer: SimpleImputer
    features: List[str]
    importances: pd.DataFrame
    metrics: Dict[str, float]
    report: str
    confusion: pd.DataFrame
    best_threshold: float
    threshold_metrics: pd.DataFrame


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
    engineered = df.copy()
    engineered["risk_classification"] = engineered["risk_classification"].astype("category")
    risk_mapping = {code: label for code, label in enumerate(engineered["risk_classification"].cat.categories)}
    engineered["risk_classification_code"] = engineered["risk_classification"].cat.codes

    engineered["hour"] = engineered["timestamp"].dt.hour
    engineered["day_of_week"] = engineered["timestamp"].dt.dayofweek
    engineered["week_of_year"] = engineered["timestamp"].dt.isocalendar().week.astype(int)
    engineered["month"] = engineered["timestamp"].dt.month
    engineered["year"] = engineered["timestamp"].dt.year
    engineered["is_weekend"] = (engineered["day_of_week"] >= 5).astype(int)
    engineered["month_sin"] = np.sin(2 * np.pi * engineered["month"] / 12.0)
    engineered["month_cos"] = np.cos(2 * np.pi * engineered["month"] / 12.0)

    return engineered, risk_mapping


def print_section(title: str) -> None:
    border = "=" * len(title)
    print(f"\n{border}\n{title}\n{border}")


def run_eda(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    eda_outputs: Dict[str, pd.DataFrame] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    print_section("Dataset Overview")
    print(f"Observations: {len(df):,}")
    print(f"Time coverage: {df['timestamp'].min()} to {df['timestamp'].max()}")

    missing = df.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        print("\nMissing value percentage (only columns with missing data):")
        print(missing.round(2).to_string())
    else:
        print("\nNo missing values detected.")
    eda_outputs["missing_values"] = missing

    print_section("Shipping Cost Distribution")
    shipping_stats = df["shipping_costs"].describe(percentiles=[0.1, 0.5, 0.9]).round(2)
    print(shipping_stats.to_string())
    eda_outputs["shipping_stats"] = shipping_stats.to_frame(name="value")

    print_section("Delivery Performance")
    delivery_stats = df["delivery_time_deviation"].describe(percentiles=[0.1, 0.5, 0.9]).round(2)
    print(delivery_stats.to_string())
    print("\nAverage delay probability by risk classification:")
    delay_by_risk = (
        df.groupby("risk_classification", observed=False)
        [["delay_probability", "delivery_time_deviation", "shipping_costs"]]
        .mean()
        .round(3)
        .sort_values("delay_probability", ascending=False)
    )
    print(delay_by_risk.to_string())
    eda_outputs["delay_by_risk"] = delay_by_risk

    print_section("Top Correlations with Shipping Costs")
    cost_corr = (
        df[numeric_cols]
        .corr()["shipping_costs"]
        .drop("shipping_costs")
        .dropna()
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .head(10)
        .round(3)
    )
    print(cost_corr.to_string())
    eda_outputs["cost_correlations"] = cost_corr.to_frame(name="correlation")

    print_section("Cost by Risk Classification")
    cost_by_risk = (
        df.groupby("risk_classification", observed=False)["shipping_costs"]
        .agg(["count", "mean", "sum"])
        .rename(columns={"count": "shipment_count", "mean": "avg_cost", "sum": "total_cost"})
        .round({"avg_cost": 2, "total_cost": 2})
        .sort_values("avg_cost", ascending=False)
    )
    print(cost_by_risk.to_string())
    eda_outputs["cost_by_risk"] = cost_by_risk

    print_section("Monthly Shipping Cost Trend")
    monthly_costs = (
        df.set_index("timestamp")
        ["shipping_costs"]
        .resample("ME").sum()
        .to_frame(name="monthly_shipping_cost")
    )
    print(monthly_costs.tail(12).round(2).to_string())
    eda_outputs["monthly_costs"] = monthly_costs

    return eda_outputs


def analyze_temporal_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    temporal_outputs: Dict[str, pd.DataFrame] = {}

    print_section("Hourly Performance")
    hourly_profile = (
        df.groupby("hour")
        [["shipping_costs", "delivery_time_deviation", "delay_probability"]]
        .agg(["mean", "count"])
    )
    hourly_profile.columns = ["_".join(col).strip() for col in hourly_profile.columns]
    hourly_profile = hourly_profile.round(3)
    print("Top 5 hours by average delay:")
    print(
        hourly_profile.sort_values("delivery_time_deviation_mean", ascending=False)
        .head(5)
        .to_string()
    )
    print("\nTop 5 hours by lowest delay:")
    print(
        hourly_profile.sort_values("delivery_time_deviation_mean", ascending=True)
        .head(5)
        .to_string()
    )
    print("\nTop 5 hours by shipping cost:")
    print(
        hourly_profile.sort_values("shipping_costs_mean", ascending=False)
        .head(5)
        .to_string()
    )
    temporal_outputs["hourly_profile"] = hourly_profile

    print_section("Day-of-Week Performance")
    dow_profile = (
        df.groupby("day_of_week")
        [["shipping_costs", "delivery_time_deviation", "delay_probability"]]
        .mean()
        .round(3)
    )
    print(dow_profile.to_string())
    temporal_outputs["day_of_week_profile"] = dow_profile

    return temporal_outputs


def create_visualizations(
    df: pd.DataFrame,
    eda_outputs: Dict[str, pd.DataFrame],
    scenarios: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    if plt is None:
        print("Matplotlib not available; skipping visualisation step.")
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if sns is not None:
        sns.set_theme(style="whitegrid")

    generated_paths: List[Path] = []

    # Shipping cost distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    if sns is not None:
        sns.histplot(df["shipping_costs"], bins=40, kde=True, color="#1f77b4", ax=ax)
    else:  # pragma: no cover - fallback
        ax.hist(df["shipping_costs"], bins=40, color="#1f77b4", alpha=0.8)
    ax.set_title("Shipping Cost Distribution")
    ax.set_xlabel("Cost per shipment ($)")
    ax.set_ylabel("Shipments")
    cost_hist_path = plots_dir / "shipping_cost_distribution.png"
    fig.tight_layout()
    fig.savefig(cost_hist_path, dpi=300)
    plt.close(fig)
    generated_paths.append(cost_hist_path)

    # Monthly shipping trend (last 18 months)
    monthly_costs = eda_outputs.get("monthly_costs")
    if monthly_costs is not None and not monthly_costs.empty:
        monthly_df = monthly_costs.reset_index().copy()
        if isinstance(monthly_df["timestamp"].dtype, pd.PeriodDtype):
            monthly_df["timestamp"] = monthly_df["timestamp"].dt.to_timestamp()
        monthly_df = monthly_df.tail(18)
        fig, ax = plt.subplots(figsize=(9, 5))
        if sns is not None:
            sns.lineplot(
                data=monthly_df,
                x="timestamp",
                y="monthly_shipping_cost",
                marker="o",
                ax=ax,
            )
        else:  # pragma: no cover
            ax.plot(
                monthly_df["timestamp"],
                monthly_df["monthly_shipping_cost"],
                marker="o",
                color="#ff7f0e",
            )
        ax.set_title("Monthly Shipping Expenditure (Last 18 Months)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total shipping cost ($)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        trend_path = plots_dir / "monthly_shipping_trend.png"
        fig.savefig(trend_path, dpi=300)
        plt.close(fig)
        generated_paths.append(trend_path)

    # Hourly delay profile
    hourly_profile = eda_outputs.get("hourly_profile")
    if hourly_profile is not None and not hourly_profile.empty:
        hourly_df = hourly_profile.reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        if sns is not None:
            sns.barplot(
                data=hourly_df,
                x="hour",
                y="delivery_time_deviation_mean",
                palette="Blues_r",
                ax=ax,
            )
        else:  # pragma: no cover
            ax.bar(hourly_df["hour"], hourly_df["delivery_time_deviation_mean"], color="#1f77b4")
        ax.set_title("Average Delivery Time Deviation by Hour")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Average deviation (hours)")
        fig.tight_layout()
        hourly_path = plots_dir / "hourly_delay_profile.png"
        fig.savefig(hourly_path, dpi=300)
        plt.close(fig)
        generated_paths.append(hourly_path)

    # Delivery deviation vs cost scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_df = df.sample(n=min(8000, len(df)), random_state=42)
    if sns is not None:
        sns.kdeplot(
            data=sample_df,
            x="delivery_time_deviation",
            y="shipping_costs",
            fill=True,
            thresh=0.05,
            levels=15,
            cmap="mako",
            ax=ax,
        )
        sns.scatterplot(
            data=sample_df,
            x="delivery_time_deviation",
            y="shipping_costs",
            s=12,
            color="#ff7f0e",
            alpha=0.3,
            ax=ax,
        )
    else:  # pragma: no cover
        ax.scatter(
            sample_df["delivery_time_deviation"],
            sample_df["shipping_costs"],
            s=12,
            alpha=0.3,
            color="#ff7f0e",
        )
    ax.set_title("Delivery Deviation vs Shipping Cost")
    ax.set_xlabel("Delivery deviation (hours)")
    ax.set_ylabel("Shipping cost ($)")
    fig.tight_layout()
    scatter_path = plots_dir / "delay_vs_cost.png"
    fig.savefig(scatter_path, dpi=300)
    plt.close(fig)
    generated_paths.append(scatter_path)

    # Route map (latitude/longitude scatter)
    fig, ax = plt.subplots(figsize=(8, 5))
    map_sample = sample_df  # reuse sample
    scatter = ax.scatter(
        map_sample["vehicle_gps_longitude"],
        map_sample["vehicle_gps_latitude"],
        c=map_sample["shipping_costs"],
        cmap="viridis",
        s=10,
        alpha=0.6,
    )
    ax.set_title("Geospatial View of Shipments (Colour = Shipping Cost)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Shipping cost ($)")
    fig.tight_layout()
    geo_path = plots_dir / "shipment_cost_map.png"
    fig.savefig(geo_path, dpi=300)
    plt.close(fig)
    generated_paths.append(geo_path)

    # Scenario savings bar chart
    if scenarios is not None and not scenarios.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        scenario_plot = scenarios.sort_values("avg_cost_reduction_per_shipment", ascending=False)
        if sns is not None:
            sns.barplot(
                data=scenario_plot,
                x="avg_cost_reduction_per_shipment",
                y="feature",
                palette="Greens",
                ax=ax,
            )
        else:  # pragma: no cover
            ax.barh(
                scenario_plot["feature"],
                scenario_plot["avg_cost_reduction_per_shipment"],
                color="#2ca02c",
            )
        ax.set_title("Simulated Savings by Operational Lever")
        ax.set_xlabel("Average cost reduction per shipment ($)")
        ax.set_ylabel("Feature adjusted")
        fig.tight_layout()
        scenario_path = plots_dir / "scenario_savings.png"
        fig.savefig(scenario_path, dpi=300)
        plt.close(fig)
        generated_paths.append(scenario_path)

    print(f"Generated {len(generated_paths)} visualisations in {plots_dir}.")
    return generated_paths


def generate_storyline(
    df: pd.DataFrame,
    eda_outputs: Dict[str, pd.DataFrame],
    cost_outputs: CostModelOutputs,
    delay_outputs: DelayModelOutputs,
    scenarios: pd.DataFrame,
) -> str:
    shipments = len(df)
    coverage_years = df["timestamp"].dt.year.nunique()
    total_spend = df["shipping_costs"].sum()
    average_cost = df["shipping_costs"].mean()

    monthly_costs = eda_outputs.get("monthly_costs")
    last_month_spend = None
    prev_quarter_avg = None
    if monthly_costs is not None and not monthly_costs.empty:
        monthly_series = monthly_costs["monthly_shipping_cost"]
        last_month_spend = float(monthly_series.iloc[-1])
        if len(monthly_series) >= 3:
            prev_quarter_avg = float(monthly_series.tail(3).mean())

    delay_threshold = df["delivery_time_deviation"].quantile(0.75)
    high_delay_rate = float((df["delivery_time_deviation"] >= delay_threshold).mean())
    avg_deviation = float(df["delivery_time_deviation"].mean())

    hourly_profile = eda_outputs.get("hourly_profile")
    busiest_hour = None
    best_hour = None
    if hourly_profile is not None and not hourly_profile.empty:
        worst_row = hourly_profile.sort_values("delivery_time_deviation_mean", ascending=False).iloc[0]
        best_row = hourly_profile.sort_values("delivery_time_deviation_mean", ascending=True).iloc[0]
        busiest_hour = (int(worst_row.name), float(worst_row["delivery_time_deviation_mean"]))
        best_hour = (int(best_row.name), float(best_row["delivery_time_deviation_mean"]))

    cost_by_risk = eda_outputs.get("cost_by_risk")
    riskiest_cost = None
    if cost_by_risk is not None and not cost_by_risk.empty:
        riskiest_row = cost_by_risk.sort_values("avg_cost", ascending=False).iloc[0]
        riskiest_cost = (
            str(riskiest_row.name),
            float(riskiest_row["avg_cost"]),
            int(riskiest_row["shipment_count"]),
        )

    top_scenario = None
    if scenarios is not None and not scenarios.empty:
        top_row = scenarios.iloc[0]
        top_scenario = (
            pretty_feature(top_row["feature"]),
            float(top_row["avg_cost_reduction_per_shipment"]),
            float(top_row["estimated_total_savings"]),
        )

    story_parts: List[str] = []
    story_parts.append(
        (
            f"GRB moved {shipments:,} shipments across {coverage_years} years of history, spending $"
            f"{total_spend:,.0f} overall (average ${average_cost:,.0f} per load)."
        )
    )

    if last_month_spend is not None and prev_quarter_avg is not None:
        change_pct = ((last_month_spend - prev_quarter_avg) / prev_quarter_avg) if prev_quarter_avg else 0.0
        story_parts.append(
            (
                f"Monthly spend recently closed at ${last_month_spend:,.0f}, "
                f"which is {change_pct * 100:.1f}% {'above' if change_pct >= 0 else 'below'} the prior-quarter average."
            )
        )

    story_parts.append(
        (
            f"Delivery performance is under pressure: the average deviation is {avg_deviation:.2f} hours, "
            f"and {high_delay_rate * 100:.1f}% of loads fall into the highest delay quartile (≥ {delay_threshold:.1f} hours)."
        )
    )

    if busiest_hour and best_hour:
        story_parts.append(
            (
                f"Dispatching at hour {busiest_hour[0]} correlates with the worst average delay ({busiest_hour[1]:.2f} hours), "
                f"whereas hour {best_hour[0]} trims lateness to {best_hour[1]:.2f} hours—pointing to a clear scheduling lever."
            )
        )

    if riskiest_cost:
        story_parts.append(
            (
                f"Risk segmentation shows {riskiest_cost[0]} lanes run the highest unit cost (${riskiest_cost[1]:.2f}) "
                f"across {riskiest_cost[2]:,} shipments, signalling priority corridors for renegotiation and process control."
            )
        )

    if top_scenario:
        story_parts.append(
            (
                f"Scenario modelling suggests that reducing {top_scenario[0]} could free up about "
                f"${top_scenario[1]:.2f} per shipment, or roughly ${top_scenario[2]:,.0f} across the network if scaled."
            )
        )

    story_parts.append(
        (
            f"Current machine-learning diagnostics flag weak predictive power (cost model R² {cost_outputs.metrics['r2_test']:.2f}; "
            f"delay ROC-AUC {delay_outputs.metrics['roc_auc']:.2f}), implying that richer operational data is needed to automate decisions."
        )
    )

    story_parts.append(
        "Taken together, the data points to a campaign that combines scheduling discipline, driver behaviour programmes, and door-to-door process tuning to bend both cost and delay curves back in line."
    )

    return "\n\n".join(story_parts)


def prepare_model_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_df = df.drop(columns=["timestamp", "risk_classification"])  # keep numeric & encoded features

    cost_target = feature_df["shipping_costs"].copy()
    delay_target = (feature_df["delivery_time_deviation"] >= feature_df["delivery_time_deviation"].quantile(0.75)).astype(int)

    feature_df = feature_df.drop(columns=["shipping_costs"])

    return feature_df, cost_target, delay_target


def train_cost_model(features: pd.DataFrame, target: pd.Series, output_dir: Path) -> CostModelOutputs:
    exclusions = {"delivery_time_deviation", "delay_probability"}
    feature_cols = [col for col in features.columns if col not in exclusions]

    X = features[feature_cols]
    y = target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=5,
    )
    model.fit(X_train_imp, y_train)

    test_pred = model.predict(X_test_imp)
    metrics = {
        "r2_test": float(r2_score(y_test, test_pred)),
        "mae_test": float(mean_absolute_error(y_test, test_pred)),
        "baseline_mean_cost": float(y_test.mean()),
    }

    importances = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.round(4).to_csv(output_dir / "cost_feature_importances.csv", index=False)

    print_section("Cost Model Performance")
    print(json.dumps(metrics, indent=2))
    print("\nTop cost drivers:")
    print(importances.head(10).to_string(index=False))

    return CostModelOutputs(model=model, imputer=imputer, features=feature_cols, importances=importances, metrics=metrics)


def train_delay_model(features: pd.DataFrame, target: pd.Series, output_dir: Path) -> DelayModelOutputs:
    exclusions = {
        "delivery_time_deviation",
        "delay_probability",
    }
    feature_cols = [col for col in features.columns if col not in exclusions]

    X = features[feature_cols]
    y = target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=5,
    )
    clf.fit(X_train_imp, y_train)

    y_prob = clf.predict_proba(X_test_imp)[:, 1]

    thresholds = np.linspace(0.25, 0.55, 16)
    threshold_records = []
    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        threshold_records.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                "positive_rate": float(preds.mean()),
            }
        )
    threshold_metrics = pd.DataFrame(threshold_records)
    best_row = threshold_metrics.sort_values("f1", ascending=False).iloc[0]
    best_threshold = float(best_row["threshold"])
    best_preds = (y_prob >= best_threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy_at_0.5": float(( (y_prob >= 0.5).astype(int) == y_test).mean()),
        "best_threshold": best_threshold,
        "best_f1": float(best_row["f1"]),
        "best_precision": float(best_row["precision"]),
        "best_recall": float(best_row["recall"]),
        "best_balanced_accuracy": float(best_row["balanced_accuracy"]),
    }

    report = classification_report(
        y_test,
        best_preds,
        target_names=["On-time", "High Delay"],
        zero_division=0,
    )
    confusion = pd.DataFrame(
        confusion_matrix(y_test, best_preds),
        index=["Actual_On-time", "Actual_High Delay"],
        columns=["Pred_On-time", "Pred_High Delay"],
    )

    importances = (
        pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.round(4).to_csv(output_dir / "delay_feature_importances.csv", index=False)
    threshold_metrics.round(4).to_csv(output_dir / "delay_threshold_tuning.csv", index=False)

    print_section("Delay Risk Model Performance")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n" + report)
    print("Confusion matrix:\n" + confusion.to_string())
    print("\nThreshold tuning (top 5 by F1):")
    print(
        threshold_metrics.sort_values("f1", ascending=False).head(5).round(3).to_string(index=False)
    )
    print("\nTop delay risk drivers:")
    print(importances.head(10).to_string(index=False))

    return DelayModelOutputs(
        model=clf,
        imputer=imputer,
        features=feature_cols,
        importances=importances,
        metrics=metrics,
        report=report,
        confusion=confusion,
        best_threshold=best_threshold,
        threshold_metrics=threshold_metrics,
    )


def simulate_cost_scenarios(
    outputs: CostModelOutputs,
    features: pd.DataFrame,
    top_features: Iterable[str],
    output_dir: Path,
) -> pd.DataFrame:
    X_all = features[list(outputs.features)]
    X_all_imp = pd.DataFrame(
        outputs.imputer.transform(X_all),
        columns=outputs.features,
        index=X_all.index,
    )
    baseline_pred = outputs.model.predict(X_all_imp)
    baseline_mean = baseline_pred.mean()

    scenario_rows = []
    for feature in top_features:
        if feature not in X_all_imp.columns:
            continue
        feature_series = X_all_imp[feature]
        high_mask = feature_series > feature_series.quantile(0.75)
        if not high_mask.any():
            continue
        scenario_frame = X_all_imp.copy()
        scenario_frame.loc[high_mask, feature] = feature_series.quantile(0.5)
        scenario_pred = outputs.model.predict(scenario_frame)
        avg_delta = baseline_mean - scenario_pred.mean()
        pct_change = avg_delta / baseline_mean if baseline_mean else 0.0
        scenario_rows.append(
            {
                "feature": feature,
                "share_adjusted": float(high_mask.mean()),
                "avg_cost_reduction_per_shipment": float(avg_delta),
                "percent_reduction_vs_baseline": float(pct_change),
                "estimated_total_savings": float(avg_delta * len(X_all_imp)),
            }
        )

    scenarios = pd.DataFrame(scenario_rows).sort_values(
        "avg_cost_reduction_per_shipment", ascending=False
    )
    scenarios.round(4).to_csv(output_dir / "scenario_cost_impact.csv", index=False)

    print_section("Scenario Simulations")
    if scenarios.empty:
        print("No valid scenarios generated.")
    else:
        print(scenarios.to_string(index=False))
    return scenarios


def derive_recommendations(
    cost_importances: pd.DataFrame,
    delay_importances: pd.DataFrame,
    scenarios: pd.DataFrame,
    eda_outputs: Dict[str, pd.DataFrame],
) -> List[str]:
    recommendations: List[str] = []

    if not scenarios.empty:
        top_row = scenarios.iloc[0]
        recommendations.append(
            (
                f"Prioritise reducing {pretty_feature(top_row['feature'])} for high-exposure lanes; a 75th-to-50th percentile shift "
                f"is modelled to save ${top_row['avg_cost_reduction_per_shipment']:.2f} per shipment "
                f"(~{top_row['percent_reduction_vs_baseline'] * 100:.1f}% reduction)."
            )
        )

    top_delay = [pretty_feature(feature) for feature in delay_importances.head(3)["feature"].tolist()]
    if top_delay:
        recommendations.append(
            "Strengthen delay mitigation across "
            + ", ".join(top_delay)
            + " via dynamic re-routing, proactive customs management, and driver coaching to lower high-delay incidence."
        )

    top_cost = [pretty_feature(feature) for feature in cost_importances.head(5)["feature"].tolist() if feature not in {"vehicle_gps_latitude", "vehicle_gps_longitude", "month", "year", "hour", "day_of_week", "week_of_year", "month_sin", "month_cos", "is_weekend"}]
    top_cost = top_cost[:3]
    if top_cost:
        recommendations.append(
            "Align inventory and planning processes to stabilise "
            + ", ".join(top_cost)
            + "; integrate these metrics into weekly S&OP dashboards to control cost volatility."
        )

    hourly_profile = eda_outputs.get("hourly_profile")
    if hourly_profile is not None and not hourly_profile.empty:
        worst_hour_row = hourly_profile.sort_values("delivery_time_deviation_mean", ascending=False).iloc[0]
        best_hour_row = hourly_profile.sort_values("delivery_time_deviation_mean", ascending=True).iloc[0]
        recommendations.append(
            (
                f"Shift dispatching away from hour {int(worst_hour_row.name)} (avg deviation {worst_hour_row['delivery_time_deviation_mean']:.2f} hrs) "
                f"toward hour {int(best_hour_row.name)} ({best_hour_row['delivery_time_deviation_mean']:.2f} hrs) to claw back schedule adherence."
            )
        )

    if "monthly_costs" in eda_outputs:
        recent_month = eda_outputs["monthly_costs"].tail(1)["monthly_shipping_cost"].values[0]
        recommendations.append(
            f"Allocate resources to peak months where shipping spend recently reached ${recent_month:,.0f}; "
            "negotiate surge capacity contracts ahead of those periods."
        )

    return recommendations


def save_summary(
    output_dir: Path,
    recommendations: List[str],
    cost_metrics: Dict[str, float],
    delay_metrics: Dict[str, float],
    storyline: str,
    visuals: Iterable[Path],
) -> None:
    summary = {
        "recommendations": recommendations,
        "cost_model_metrics": cost_metrics,
        "delay_model_metrics": delay_metrics,
        "storyline": storyline,
        "visualisations": [str(path) for path in visuals],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    data_path = Path("dynamic_supply_chain_logistics_dataset.csv")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    df_raw = load_dataset(data_path)
    df_engineered, risk_mapping = engineer_features(df_raw)

    print_section("Risk Classification Mapping")
    print(risk_mapping)

    eda_outputs = run_eda(df_engineered)
    temporal_outputs = analyze_temporal_patterns(df_engineered)
    eda_outputs.update(temporal_outputs)

    feature_df, cost_target, delay_target = prepare_model_frames(df_engineered)

    cost_outputs = train_cost_model(feature_df, cost_target, output_dir)
    delay_outputs = train_delay_model(feature_df, delay_target, output_dir)

    excluded_for_actions = {
        "vehicle_gps_latitude",
        "vehicle_gps_longitude",
        "hour",
        "day_of_week",
        "week_of_year",
        "month",
        "year",
        "month_sin",
        "month_cos",
        "is_weekend",
    }
    top_cost_features = [
        feat
        for feat in cost_outputs.importances["feature"].tolist()
        if feat not in excluded_for_actions
    ][:5]
    scenarios = simulate_cost_scenarios(cost_outputs, feature_df, top_cost_features, output_dir)

    recommendations = derive_recommendations(
        cost_outputs.importances,
        delay_outputs.importances,
        scenarios,
        eda_outputs,
    )

    storyline = generate_storyline(
        df_engineered,
        eda_outputs,
        cost_outputs,
        delay_outputs,
        scenarios,
    )

    print_section("Actionable Recommendations")
    for idx, rec in enumerate(recommendations, start=1):
        print(f"{idx}. {rec}")

    print_section("Data Story")
    print(storyline)

    visualisations = create_visualizations(df_engineered, eda_outputs, scenarios, output_dir)

    save_summary(
        output_dir,
        recommendations,
        cost_outputs.metrics,
        delay_outputs.metrics,
        storyline,
        visualisations,
    )


if __name__ == "__main__":
    main()
