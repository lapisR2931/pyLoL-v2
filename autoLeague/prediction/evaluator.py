"""
Phase 6: モデル評価・比較モジュール

視界スコアの貢献度を評価する。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)


# =============================================================================
# モデル評価
# =============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict:
    """
    モデル評価

    Args:
        y_true: 正解ラベル (N,)
        y_pred: 予測ラベル (N,)
        y_proba: 予測確率 (N,) - オプション

    Returns:
        評価メトリクス辞書
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # 混同行列
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # AUC, Log Loss（確率が与えられた場合）
    if y_proba is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["auc"] = 0.5

        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except ValueError:
            metrics["log_loss"] = float('inf')

    return metrics


# =============================================================================
# モデル比較
# =============================================================================

def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    モデル比較表を生成

    Args:
        results: {model_name: metrics_dict} の辞書

    Returns:
        比較DataFrameの文字列表現
    """
    rows = []

    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "CV Accuracy": metrics.get("cv_accuracy", metrics.get("accuracy", 0)),
            "CV AUC": metrics.get("cv_auc", metrics.get("auc", 0)),
            "CV LogLoss": metrics.get("cv_log_loss", metrics.get("log_loss", 0)),
            "N Features": metrics.get("n_features", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def calculate_contribution(
    baseline_metrics: Dict,
    extended_metrics: Dict,
    metric_key: str = "cv_accuracy",
) -> Dict:
    """
    視界スコアの貢献度を計算

    貢献度 = (拡張モデル精度 - ベースライン精度) / (1 - ベースライン精度) * 100%

    Args:
        baseline_metrics: ベースラインモデルのメトリクス
        extended_metrics: 拡張モデルのメトリクス
        metric_key: 比較する指標キー

    Returns:
        貢献度情報
    """
    baseline_score = baseline_metrics.get(metric_key, 0)
    extended_score = extended_metrics.get(metric_key, 0)

    # 絶対的な向上
    absolute_lift = extended_score - baseline_score

    # 相対的な貢献度（改善余地に対する割合）
    if baseline_score < 1.0:
        relative_contribution = absolute_lift / (1.0 - baseline_score)
    else:
        relative_contribution = 0.0

    return {
        "baseline_score": baseline_score,
        "extended_score": extended_score,
        "absolute_lift": absolute_lift,
        "absolute_lift_pct": absolute_lift * 100,
        "relative_contribution": relative_contribution,
        "relative_contribution_pct": relative_contribution * 100,
        "metric": metric_key,
    }


def summarize_contributions(
    results: Dict[str, Dict],
    baseline_key: str = "baseline",
) -> Dict[str, Dict]:
    """
    各拡張モデルの貢献度を計算

    Args:
        results: {model_name: metrics_dict} の辞書
        baseline_key: ベースラインモデルのキー

    Returns:
        {extended_model_name: contribution_dict}
    """
    baseline_metrics = results.get(baseline_key, {})
    contributions = {}

    for model_name, metrics in results.items():
        if model_name == baseline_key:
            continue

        contribution = calculate_contribution(baseline_metrics, metrics)
        contribution["model_name"] = model_name
        contributions[model_name] = contribution

    return contributions


# =============================================================================
# レポート生成
# =============================================================================

def generate_report(
    results_10min: Dict[str, Dict],
    results_20min: Dict[str, Dict],
    output_path: Optional[Path] = None,
) -> str:
    """
    比較レポートを生成

    Args:
        results_10min: 10分時点の結果
        results_20min: 20分時点の結果
        output_path: JSON出力先パス（オプション）

    Returns:
        レポート文字列
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Phase 6: 勝敗予測モデル比較レポート")
    lines.append("=" * 60)

    # 10分時点
    lines.append("\n## 10分時点")
    lines.append("-" * 40)

    df_10 = compare_models(results_10min)
    lines.append(df_10.to_string(index=False))

    contributions_10 = summarize_contributions(results_10min)
    lines.append("\n### 視界スコア貢献度 (10分)")
    for model_name, contrib in contributions_10.items():
        lines.append(f"  {model_name}:")
        lines.append(f"    精度向上: {contrib['absolute_lift_pct']:.2f}%")
        lines.append(f"    相対貢献度: {contrib['relative_contribution_pct']:.2f}%")

    # 20分時点
    lines.append("\n## 20分時点")
    lines.append("-" * 40)

    df_20 = compare_models(results_20min)
    lines.append(df_20.to_string(index=False))

    contributions_20 = summarize_contributions(results_20min)
    lines.append("\n### 視界スコア貢献度 (20分)")
    for model_name, contrib in contributions_20.items():
        lines.append(f"  {model_name}:")
        lines.append(f"    精度向上: {contrib['absolute_lift_pct']:.2f}%")
        lines.append(f"    相対貢献度: {contrib['relative_contribution_pct']:.2f}%")

    # 結論
    lines.append("\n" + "=" * 60)
    lines.append("## 結論")
    lines.append("-" * 40)

    # 最良モデルを特定
    all_results = {
        "10min": results_10min,
        "20min": results_20min,
    }

    best_model = None
    best_acc = 0
    best_time = None

    for time_name, time_results in all_results.items():
        for model_name, metrics in time_results.items():
            acc = metrics.get("cv_accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_model = model_name
                best_time = time_name

    if best_model:
        lines.append(f"最高精度: {best_model} @ {best_time} (Accuracy: {best_acc:.3f})")

    # モデル比較（baseline以外）
    for time_name in ["10min", "20min"]:
        time_results = all_results[time_name]
        baseline_acc = time_results.get("baseline", {}).get("cv_accuracy", 0)

        # 各モデルの精度向上を計算
        model_lifts = []
        for model_name in ["baseline_riot", "baseline_grid", "baseline_tactical"]:
            if model_name in time_results:
                acc = time_results[model_name].get("cv_accuracy", 0)
                lift = (acc - baseline_acc) * 100
                model_lifts.append((model_name, acc, lift))

        if model_lifts:
            # 最も精度向上が大きいモデルを特定
            model_lifts.sort(key=lambda x: x[2], reverse=True)
            best_model, best_acc, best_lift = model_lifts[0]
            lines.append(f"{time_name}: {best_model} が +{best_lift:.2f}% で最良")

    report = "\n".join(lines)

    # JSON出力
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_data = {
            "10min": {
                "results": results_10min,
                "contributions": contributions_10,
            },
            "20min": {
                "results": results_20min,
                "contributions": contributions_20,
            },
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"JSON保存: {output_path}")

    return report


# =============================================================================
# 可視化
# =============================================================================

def visualize_comparison(
    results_10min: Dict[str, Dict],
    results_20min: Dict[str, Dict],
    output_dir: Optional[Path] = None,
) -> None:
    """
    モデル比較を可視化

    Args:
        results_10min: 10分時点の結果
        results_20min: 20分時点の結果
        output_dir: 画像出力先ディレクトリ
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib がインストールされていません。可視化をスキップします。")
        return

    # 日本語フォント設定
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (time_name, results) in enumerate([("10min", results_10min), ("20min", results_20min)]):
        ax = axes[idx]

        model_names = list(results.keys())
        accuracies = [results[m].get("cv_accuracy", 0) for m in model_names]
        aucs = [results[m].get("cv_auc", 0) for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
        bars2 = ax.bar(x + width/2, aucs, width, label='AUC', color='darkorange')

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(f'Model Comparison @ {time_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)

        # 値を表示
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"画像保存: {output_path}")

    plt.show()


def visualize_feature_importance(
    results: Dict[str, Dict],
    time_name: str,
    output_dir: Optional[Path] = None,
) -> None:
    """
    特徴量重要度を可視化

    Args:
        results: モデル結果
        time_name: 時間帯名（"10min" or "20min"）
        output_dir: 画像出力先ディレクトリ
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib がインストールされていません。可視化をスキップします。")
        return

    n_models = len(results)
    n_cols = min(n_models, 4)  # 最大4列
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))

    # axesを1次元配列に統一
    if n_models == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, (model_name, metrics) in enumerate(results.items()):
        ax = axes[idx]

        importance = metrics.get("feature_importance", [])
        if not importance:
            ax.set_title(f'{model_name} - No importance data')
            continue

        names = [item[0] for item in importance]
        values = [item[1] for item in importance]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (|coef|)')
        ax.set_title(f'{model_name} @ {time_name}')

    # 余った軸を非表示
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"feature_importance_{time_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"画像保存: {output_path}")

    plt.show()


# =============================================================================
# 日本語グラフ生成
# =============================================================================

def setup_japanese_font() -> str:
    """
    Windows環境で日本語フォントを設定

    Returns:
        使用するフォント名
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Windowsで利用可能な日本語フォント（優先順）
    japanese_fonts = [
        'Yu Gothic',      # Windows 10以降
        'Meiryo',         # Windows Vista以降
        'MS Gothic',      # 従来のWindowsフォント
        'Hiragino Sans',  # macOS（互換性のため）
    ]

    available_fonts = set(f.name for f in fm.fontManager.ttflist)

    for font in japanese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False  # マイナス記号対策
            return font

    # フォントが見つからない場合は警告
    print("警告: 日本語フォントが見つかりません。英語フォントを使用します。")
    return 'sans-serif'


def visualize_accuracy_over_time_ja(
    results_by_time: Dict[str, Dict[str, Dict]],
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    時間帯別の精度推移を日本語折れ線グラフで表示

    Args:
        results_by_time: {
            "5min": {"baseline": {...}, "baseline_riot": {...}, ...},
            "10min": {...},
            ...
        }
        output_path: 出力先パス
        figsize: 図のサイズ
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib がインストールされていません。可視化をスキップします。")
        return

    from .config import MODEL_NAMES_JA, MODEL_COLORS

    # 日本語フォント設定
    font_name = setup_japanese_font()

    # データ準備
    time_names = list(results_by_time.keys())
    time_labels = []
    for t in time_names:
        # "5min" -> "5分" に変換
        mins = t.replace("min", "")
        time_labels.append(f"{mins}分")

    model_keys = ["baseline", "baseline_riot", "baseline_grid", "baseline_tactical"]

    fig, ax = plt.subplots(figsize=figsize)

    for model_key in model_keys:
        accuracies = []
        for time_name in time_names:
            if model_key in results_by_time.get(time_name, {}):
                acc = results_by_time[time_name][model_key].get("cv_accuracy", 0)
                accuracies.append(acc * 100)  # パーセント表示
            else:
                accuracies.append(None)

        label = MODEL_NAMES_JA.get(model_key, model_key)
        color = MODEL_COLORS.get(model_key, "#333333")

        ax.plot(time_labels, accuracies, marker='o', label=label, color=color, linewidth=2, markersize=8)

        # 各点に値を表示
        for i, acc in enumerate(accuracies):
            if acc is not None:
                ax.annotate(f'{acc:.1f}%',
                           xy=(time_labels[i], acc),
                           xytext=(0, 8),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('評価時点', fontsize=12)
    ax.set_ylabel('予測精度 (%)', fontsize=12)
    ax.set_title('時間帯別 勝敗予測精度の推移', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 100)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"日本語グラフ保存: {output_path}")

    plt.show()


def visualize_contribution_over_time_ja(
    results_by_time: Dict[str, Dict[str, Dict]],
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    視界スコア追加による精度向上量を時点別に日本語グラフで表示

    Args:
        results_by_time: 時点ごとの結果辞書
        output_path: 出力先パス
        figsize: 図のサイズ
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib がインストールされていません。可視化をスキップします。")
        return

    from .config import MODEL_NAMES_JA, MODEL_COLORS

    # 日本語フォント設定
    font_name = setup_japanese_font()

    # データ準備
    time_names = list(results_by_time.keys())
    time_labels = []
    for t in time_names:
        mins = t.replace("min", "")
        time_labels.append(f"{mins}分")

    # ベースラインからの精度向上を計算
    model_keys = ["baseline_riot", "baseline_tactical"]  # 主要モデルのみ

    fig, ax = plt.subplots(figsize=figsize)

    for model_key in model_keys:
        lifts = []
        for time_name in time_names:
            baseline_acc = results_by_time.get(time_name, {}).get("baseline", {}).get("cv_accuracy", 0)
            model_acc = results_by_time.get(time_name, {}).get(model_key, {}).get("cv_accuracy", 0)
            lift = (model_acc - baseline_acc) * 100  # ポイント差
            lifts.append(lift)

        label = MODEL_NAMES_JA.get(model_key, model_key)
        color = MODEL_COLORS.get(model_key, "#333333")

        ax.plot(time_labels, lifts, marker='s', label=label, color=color, linewidth=2, markersize=8)

        # 各点に値を表示
        for i, lift in enumerate(lifts):
            ax.annotate(f'+{lift:.1f}pt',
                       xy=(time_labels[i], lift),
                       xytext=(0, 8),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('評価時点', fontsize=12)
    ax.set_ylabel('精度向上 (ポイント)', fontsize=12)
    ax.set_title('視界スコア追加による精度向上の推移', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"日本語グラフ保存: {output_path}")

    plt.show()


def generate_report_multi_time(
    results_by_time: Dict[str, Dict[str, Dict]],
    output_path: Optional[Path] = None,
) -> str:
    """
    複数時点対応のレポート生成

    Args:
        results_by_time: {time_name: {model_name: metrics_dict}}
        output_path: JSON出力先パス（オプション）

    Returns:
        レポート文字列
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Phase 6: 勝敗予測モデル比較レポート（5分刻み）")
    lines.append("=" * 60)

    all_contributions = {}

    for time_name, results in results_by_time.items():
        lines.append(f"\n## {time_name}時点")
        lines.append("-" * 40)

        df = compare_models(results)
        lines.append(df.to_string(index=False))

        contributions = summarize_contributions(results)
        all_contributions[time_name] = contributions

        lines.append(f"\n### 視界スコア貢献度 ({time_name})")
        for model_name, contrib in contributions.items():
            lines.append(f"  {model_name}:")
            lines.append(f"    精度向上: {contrib['absolute_lift_pct']:.2f}%")
            lines.append(f"    相対貢献度: {contrib['relative_contribution_pct']:.2f}%")

    # 結論
    lines.append("\n" + "=" * 60)
    lines.append("## 結論")
    lines.append("-" * 40)

    # 最良モデルを特定
    best_model = None
    best_acc = 0
    best_time = None

    for time_name, results in results_by_time.items():
        for model_name, metrics in results.items():
            acc = metrics.get("cv_accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_model = model_name
                best_time = time_name

    if best_model:
        lines.append(f"最高精度: {best_model} @ {best_time} (Accuracy: {best_acc:.3f})")

    report = "\n".join(lines)

    # JSON出力
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_data = {
            time_name: {
                "results": results,
                "contributions": all_contributions.get(time_name, {}),
            }
            for time_name, results in results_by_time.items()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"JSON保存: {output_path}")

    return report
