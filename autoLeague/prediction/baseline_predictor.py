"""
Phase 6: 勝敗予測モデル

ベースライン特徴量 + 視界スコア特徴量から勝敗を予測するモデル。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from .config import LOGISTIC_C, LOGISTIC_MAX_ITER


# =============================================================================
# 勝敗予測モデル
# =============================================================================

class WinPredictor:
    """
    勝敗予測モデル

    feature_setで使用する特徴量を指定:
    - "baseline": ベースライン特徴量のみ
    - "baseline_riot": ベースライン + Riot visionScore推定値
    - "baseline_grid": ベースライン + 自作視界スコア（グリッド→スカラー）
    - "baseline_tactical": ベースライン + 戦術スコア（6特徴量）

    使用例:
        predictor = WinPredictor(feature_set="baseline")
        metrics = predictor.fit(X_baseline, y)
        y_pred = predictor.predict(X_baseline_test)
    """

    VALID_FEATURE_SETS = ("baseline", "baseline_riot", "baseline_grid", "baseline_tactical")

    def __init__(self, feature_set: str = "baseline"):
        """
        Args:
            feature_set: 特徴量セット（"baseline", "baseline_riot", "baseline_grid", "baseline_tactical"）
        """
        if feature_set not in self.VALID_FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {self.VALID_FEATURE_SETS}, got {feature_set}")

        self.feature_set = feature_set
        self.model = None
        self.scaler = None
        self._is_fitted = False
        self._train_metrics = {}
        self._feature_names = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        モデルを学習

        Args:
            X: 入力データ (N, F)
            y: ラベル (N,)
            feature_names: 特徴量名リスト
            verbose: 進捗表示

        Returns:
            学習メトリクス
        """
        if feature_names is not None:
            self._feature_names = feature_names

        # 正規化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # モデル構築
        self.model = LogisticRegression(
            C=LOGISTIC_C,
            max_iter=LOGISTIC_MAX_ITER,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs',
        )

        # 学習
        self.model.fit(X_scaled, y)

        # 学習データでの精度
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        train_acc = accuracy_score(y, y_pred)

        try:
            train_auc = roc_auc_score(y, y_proba)
        except ValueError:
            train_auc = 0.5  # 1クラスのみの場合

        metrics = {
            "train_accuracy": float(train_acc),
            "train_auc": float(train_auc),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_set": self.feature_set,
        }

        if verbose:
            print(f"[{self.feature_set}] Train Accuracy: {train_acc:.3f}, AUC: {train_auc:.3f}")

        self._is_fitted = True
        self._train_metrics = metrics

        return metrics

    def fit_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Leave-One-Out交差検証で学習・評価

        Args:
            X: 入力データ (N, F)
            y: ラベル (N,)
            feature_names: 特徴量名リスト
            verbose: 進捗表示

        Returns:
            交差検証メトリクス
        """
        if feature_names is not None:
            self._feature_names = feature_names

        # 正規化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # モデル構築
        self.model = LogisticRegression(
            C=LOGISTIC_C,
            max_iter=LOGISTIC_MAX_ITER,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs',
        )

        # LOO交差検証
        loo = LeaveOneOut()

        # 予測
        y_pred_cv = cross_val_predict(self.model, X_scaled, y, cv=loo)
        y_proba_cv = cross_val_predict(self.model, X_scaled, y, cv=loo, method='predict_proba')[:, 1]

        # 全データで最終学習
        self.model.fit(X_scaled, y)

        # 評価指標
        cv_acc = accuracy_score(y, y_pred_cv)

        try:
            cv_auc = roc_auc_score(y, y_proba_cv)
        except ValueError:
            cv_auc = 0.5

        try:
            cv_logloss = log_loss(y, y_proba_cv)
        except ValueError:
            cv_logloss = float('inf')

        metrics = {
            "cv_accuracy": float(cv_acc),
            "cv_auc": float(cv_auc),
            "cv_log_loss": float(cv_logloss),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_set": self.feature_set,
            "y_pred_cv": y_pred_cv.tolist(),
            "y_proba_cv": y_proba_cv.tolist(),
        }

        if verbose:
            print(f"[{self.feature_set}] LOO-CV Accuracy: {cv_acc:.3f}, AUC: {cv_auc:.3f}, LogLoss: {cv_logloss:.3f}")

        self._is_fitted = True
        self._train_metrics = metrics

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測

        Args:
            X: 入力データ (N, F)

        Returns:
            予測ラベル (N,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測

        Args:
            X: 入力データ (N, F)

        Returns:
            Blue勝利確率 (N,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        """
        特徴量重要度を取得

        Returns:
            係数の絶対値 (F,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return np.abs(self.model.coef_[0])

    def get_feature_importance_with_names(self) -> List[Tuple[str, float]]:
        """
        特徴量名付きで重要度を取得

        Returns:
            [(feature_name, importance), ...] 重要度降順
        """
        importance = self.get_feature_importance()

        if self._feature_names:
            pairs = list(zip(self._feature_names, importance))
        else:
            pairs = [(f"feature_{i}", imp) for i, imp in enumerate(importance)]

        return sorted(pairs, key=lambda x: x[1], reverse=True)

    def save(self, path: Union[str, Path]) -> None:
        """
        モデルを保存

        Args:
            path: 保存先パス
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "feature_set": self.feature_set,
            "model": self.model,
            "scaler": self.scaler,
            "train_metrics": self._train_metrics,
            "feature_names": self._feature_names,
        }

        joblib.dump(save_dict, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WinPredictor":
        """
        モデルを読み込み

        Args:
            path: モデルファイルパス

        Returns:
            WinPredictorインスタンス
        """
        path = Path(path)
        save_dict = joblib.load(path)

        predictor = cls(feature_set=save_dict["feature_set"])
        predictor.model = save_dict["model"]
        predictor.scaler = save_dict["scaler"]
        predictor._train_metrics = save_dict["train_metrics"]
        predictor._feature_names = save_dict["feature_names"]
        predictor._is_fitted = True

        return predictor


# =============================================================================
# ユーティリティ関数
# =============================================================================

def prepare_features(
    dataset: Dict,
    time_index: int,
    feature_set: str,
    vision_predictor=None,
) -> Tuple[np.ndarray, List[str]]:
    """
    データセットから指定時点・特徴量セットの特徴量を準備

    Args:
        dataset: load_prediction_datasetの出力
        time_index: 時点インデックス（0=10分, 1=20分）
        feature_set: "baseline", "baseline_riot", "baseline_grid"
        vision_predictor: Phase 5のVisionPredictor（feature_set="baseline_grid"時に使用）

    Returns:
        (X, feature_names)
    """
    X_baseline = dataset["X_baseline"][:, time_index, :]  # (N, F_base)
    X_riot_vision = dataset["X_riot_vision"][:, time_index, :]  # (N, 3)
    X_ward_grid = dataset["X_ward_grid"][:, time_index, :, :, :]  # (N, 2, 32, 32)

    baseline_features = dataset["baseline_features"]
    riot_vision_features = dataset["riot_vision_features"]

    if feature_set == "baseline":
        X = X_baseline
        feature_names = list(baseline_features)

    elif feature_set == "baseline_riot":
        X = np.concatenate([X_baseline, X_riot_vision], axis=1)
        feature_names = list(baseline_features) + list(riot_vision_features)

    elif feature_set == "baseline_grid":
        # ward gridからスカラー特徴量を生成
        if vision_predictor is not None:
            # Phase 5のVisionPredictorを使用
            # VisionPredictorは (N, 2, 3, 32, 32) を期待するため、
            # 時間帯次元を追加: (N, 2, 32, 32) -> (N, 2, 1, 32, 32)
            X_grid_expanded = X_ward_grid[:, :, np.newaxis, :, :]  # (N, 2, 1, 32, 32)

            # 3時間帯分にパディング（同じデータを繰り返す）
            X_grid_padded = np.repeat(X_grid_expanded, 3, axis=2)  # (N, 2, 3, 32, 32)

            vision_score = vision_predictor.predict_proba(X_grid_padded)  # (N,)
            vision_score = vision_score.reshape(-1, 1)  # (N, 1)
        else:
            # 簡易版: グリッドの要約統計量を使用
            # Blue ward密度 - Red ward密度
            blue_density = (X_ward_grid[:, 0, :, :] > 0).sum(axis=(1, 2))  # (N,)
            red_density = (X_ward_grid[:, 1, :, :] > 0).sum(axis=(1, 2))  # (N,)
            blue_total = X_ward_grid[:, 0, :, :].sum(axis=(1, 2))  # (N,)
            red_total = X_ward_grid[:, 1, :, :].sum(axis=(1, 2))  # (N,)

            vision_score = np.stack([
                blue_density,
                red_density,
                blue_density - red_density,
                blue_total,
                red_total,
                blue_total - red_total,
            ], axis=1)  # (N, 6)

        X = np.concatenate([X_baseline, vision_score], axis=1)
        feature_names = list(baseline_features) + [
            "grid_blue_density", "grid_red_density", "grid_density_diff",
            "grid_blue_total", "grid_red_total", "grid_total_diff",
        ]

    elif feature_set == "baseline_tactical":
        # 戦術スコア特徴量を追加
        if "X_tactical" not in dataset:
            raise ValueError("X_tactical not found in dataset. Rebuild dataset with tactical scores.")

        X_tactical = dataset["X_tactical"][:, time_index, :]  # (N, 6)
        tactical_features = dataset.get("tactical_features", [
            "blue_placement_score", "red_placement_score",
            "blue_deny_score", "red_deny_score",
            "placement_score_diff", "deny_score_diff",
        ])

        X = np.concatenate([X_baseline, X_tactical], axis=1)
        feature_names = list(baseline_features) + list(tactical_features)

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    return X, feature_names


def train_all_models(
    dataset: Dict,
    time_index: int,
    output_dir: Optional[Path] = None,
    vision_predictor=None,
    verbose: bool = True,
    include_tactical: bool = True,
) -> Dict[str, Dict]:
    """
    全モデルを学習

    Args:
        dataset: load_prediction_datasetの出力
        time_index: 時点インデックス（0=10分, 1=20分）
        output_dir: モデル保存先ディレクトリ
        vision_predictor: Phase 5のVisionPredictor
        verbose: 進捗表示
        include_tactical: baseline_tacticalを含めるか

    Returns:
        {
            "baseline": {"cv_accuracy": float, ...},
            "baseline_riot": {...},
            "baseline_grid": {...},
            "baseline_tactical": {...},
        }
    """
    y = dataset["y"]
    results = {}

    feature_sets = ["baseline", "baseline_riot", "baseline_grid"]
    if include_tactical and "X_tactical" in dataset:
        feature_sets.append("baseline_tactical")

    for feature_set in feature_sets:
        X, feature_names = prepare_features(
            dataset, time_index, feature_set, vision_predictor
        )

        predictor = WinPredictor(feature_set=feature_set)
        metrics = predictor.fit_with_cv(X, y, feature_names=feature_names, verbose=verbose)

        # 特徴量重要度を追加
        importance = predictor.get_feature_importance_with_names()
        metrics["feature_importance"] = importance[:10]  # TOP10

        results[feature_set] = metrics

        # モデル保存
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            time_name = dataset["times_ms"][time_index] // 60000
            model_path = output_dir / f"{feature_set}_{time_name}min.joblib"
            predictor.save(model_path)
            if verbose:
                print(f"モデル保存: {model_path}")

    return results
