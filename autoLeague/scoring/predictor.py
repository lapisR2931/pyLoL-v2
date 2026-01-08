"""
視界スコア予測モデル - Phase 5 Task C

グリッド特徴量から勝敗を予測するモデル。
- ロジスティック回帰（ベースライン）
- 浅いCNN（発展版）

入力形式:
    X: (N, 2, 7, 32, 32) - N試合, 2チーム(Blue/Red), 7時間帯(5分刻み), 32x32グリッド
    y: (N,) - 勝敗ラベル (1=Blue勝利, 0=Red勝利)

出力:
    - 予測ラベル
    - 特徴量重要度 (7, 32, 32)
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 設定
# =============================================================================

# グリッド設定
GRID_SIZE = 32
N_TEAMS = 2
N_TIME_PHASES = 7  # 5分刻み: 0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30+
FEATURE_DIM = N_TEAMS * N_TIME_PHASES * GRID_SIZE * GRID_SIZE  # 14336

# ロジスティック回帰設定
LOGISTIC_C = 0.1  # 正則化強度（小さいほど強い正則化）
LOGISTIC_MAX_ITER = 1000

# CNN設定
CNN_LEARNING_RATE = 0.001
CNN_BATCH_SIZE = 8
CNN_DEFAULT_EPOCHS = 50
CNN_PATIENCE = 10  # 早期停止


# =============================================================================
# CNN モデル定義
# =============================================================================

class ShallowCNN(nn.Module):
    """
    浅い畳み込みニューラルネットワーク

    過学習対策として極めて浅い構造:
    - Conv2d x 2層
    - Global Average Pooling
    - 全結合層 x 1
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        # 入力: (N, 14, 32, 32) - 2チーム x 7時間帯 を結合
        n_input_channels = N_TEAMS * N_TIME_PHASES  # 14
        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout * 0.5),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Grad-CAM用にフック登録
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        """勾配を保存するフック"""
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 2, 3, 32, 32)
        Returns:
            (N, 1) 確率
        """
        # (N, 2, 3, 32, 32) -> (N, 6, 32, 32)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, GRID_SIZE, GRID_SIZE)

        # 特徴抽出
        feat = self.features(x)

        # Grad-CAM用に活性化を保存（推論時のみ、フックは外部で管理）
        self.activations = feat

        # 分類
        pooled = self.gap(feat).view(batch_size, -1)
        out = self.classifier(pooled)
        return torch.sigmoid(out)


# =============================================================================
# VisionPredictor クラス
# =============================================================================

class VisionPredictor:
    """
    視界特徴量から勝敗を予測するモデル

    使用例:
        predictor = VisionPredictor(model_type="logistic")
        metrics = predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)
        importance = predictor.get_feature_importance()
        predictor.save(Path("models/vision_predictor.joblib"))
    """

    def __init__(self, model_type: str = "logistic"):
        """
        Args:
            model_type: "logistic" or "cnn"
        """
        if model_type not in ("logistic", "cnn"):
            raise ValueError(f"model_type must be 'logistic' or 'cnn', got {model_type}")

        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_fitted = False
        self._train_metrics = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = CNN_DEFAULT_EPOCHS,
        verbose: bool = True
    ) -> Dict:
        """
        モデルを学習

        Args:
            X: 入力データ (N, 2, 3, 32, 32)
            y: ラベル (N,)
            epochs: CNNのエポック数
            verbose: 進捗表示

        Returns:
            学習メトリクス
        """
        start_time = time.time()

        if self.model_type == "logistic":
            metrics = self._fit_logistic(X, y, verbose)
        else:
            metrics = self._fit_cnn(X, y, epochs, verbose)

        metrics["training_time_sec"] = round(time.time() - start_time, 2)
        metrics["model_type"] = self.model_type
        metrics["n_samples"] = len(X)

        self._is_fitted = True
        self._train_metrics = metrics

        return metrics

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray, verbose: bool) -> Dict:
        """ロジスティック回帰の学習"""
        # Flatten: (N, 2, 3, 32, 32) -> (N, 6144)
        X_flat = X.reshape(len(X), -1)

        # 正規化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flat)

        # モデル構築・学習
        self.model = LogisticRegression(
            C=LOGISTIC_C,
            max_iter=LOGISTIC_MAX_ITER,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs',
            verbose=1 if verbose else 0
        )

        self.model.fit(X_scaled, y)

        # 学習データでの精度
        y_pred = self.model.predict(X_scaled)
        train_acc = accuracy_score(y, y_pred)

        if verbose:
            print(f"Train accuracy: {train_acc:.3f}")

        return {
            "train_accuracy": float(train_acc),
            "regularization_C": LOGISTIC_C,
        }

    def _fit_cnn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        verbose: bool
    ) -> Dict:
        """CNNの学習"""
        # データ変換
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=CNN_BATCH_SIZE, shuffle=True)

        # モデル構築
        self.model = ShallowCNN().to(self.device)

        # クラス重み計算（不均衡対策）
        pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=CNN_LEARNING_RATE)

        # 学習ループ
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # forward (sigmoidはモデル内で適用済み)
                output = self.model(batch_X)
                # BCEWithLogitsLoss用にsigmoid前の値が必要なので調整
                # ただしShallowCNNはsigmoid適用済みなのでBCELossを使用
                loss = nn.BCELoss()(output, batch_y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            # 早期停止判定
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            if patience_counter >= CNN_PATIENCE:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # 学習データでの精度
        self.model.eval()
        with torch.no_grad():
            y_pred = (self.model(X_tensor) > 0.5).float().cpu().numpy().flatten()
        train_acc = accuracy_score(y, y_pred)

        if verbose:
            print(f"Train accuracy: {train_acc:.3f}")

        return {
            "train_accuracy": float(train_acc),
            "epochs_trained": epoch + 1,
            "final_loss": float(avg_loss),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測

        Args:
            X: 入力データ (N, 2, 3, 32, 32)

        Returns:
            予測ラベル (N,)
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測

        Args:
            X: 入力データ (N, 2, 3, 32, 32)

        Returns:
            Blue勝利確率 (N,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.model_type == "logistic":
            X_flat = X.reshape(len(X), -1)
            X_scaled = self.scaler.transform(X_flat)
            proba = self.model.predict_proba(X_scaled)[:, 1]
        else:
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                proba = self.model(X_tensor).cpu().numpy().flatten()

        return proba

    def get_feature_importance(self) -> np.ndarray:
        """
        特徴量重要度を取得

        Returns:
            重要度 (3, 32, 32)
            - 正の値: Blue有利に寄与
            - 負の値: Red有利に寄与
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.model_type == "logistic":
            return self._get_importance_logistic()
        else:
            return self._get_importance_cnn()

    def _get_importance_logistic(self) -> np.ndarray:
        """ロジスティック回帰の特徴量重要度"""
        # 係数: (6144,) -> (2, 3, 32, 32)
        coef = self.model.coef_[0]
        coef_reshaped = coef.reshape(N_TEAMS, N_TIME_PHASES, GRID_SIZE, GRID_SIZE)

        # Blue係数 - Red係数
        # 正の値 = Blueのward存在がBlue勝利に寄与
        # 負の値 = Redのward存在がRed勝利に寄与
        importance = coef_reshaped[0] - coef_reshaped[1]  # (3, 32, 32)

        return importance.astype(np.float32)

    def _get_importance_cnn(self) -> np.ndarray:
        """
        CNNの特徴量重要度（簡易版）

        最終畳み込み層の重みを使用した近似
        """
        # 最終conv層の重みを取得
        # features[4] = 2番目のConv2d
        conv_weight = self.model.features[4].weight.data.cpu().numpy()  # (32, 16, 3, 3)

        # チャンネル方向に平均して重要度マップを生成
        # 簡易実装: 全フィルタの平均絶対値
        importance_flat = np.abs(conv_weight).mean(axis=(0, 2, 3))  # (16,)

        # 入力チャンネル(6)に対応する重みを取得
        first_conv_weight = self.model.features[0].weight.data.cpu().numpy()  # (16, 6, 3, 3)
        channel_importance = np.abs(first_conv_weight).mean(axis=(0, 2, 3))  # (6,)

        # (6,) -> (2, 3) -> Blue-Redの差分 -> (3,)
        channel_reshaped = channel_importance.reshape(N_TEAMS, N_TIME_PHASES)
        time_importance = channel_reshaped[0] - channel_reshaped[1]  # (3,)

        # 各時間帯に対して32x32のマップを生成（均一）
        importance = np.zeros((N_TIME_PHASES, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for t in range(N_TIME_PHASES):
            importance[t] = time_importance[t]

        return importance

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

        if self.model_type == "logistic":
            # joblib形式で保存
            save_dict = {
                "model_type": self.model_type,
                "model": self.model,
                "scaler": self.scaler,
                "train_metrics": self._train_metrics,
            }
            joblib.dump(save_dict, path)
        else:
            # PyTorch形式で保存
            save_dict = {
                "model_type": self.model_type,
                "model_state_dict": self.model.state_dict(),
                "train_metrics": self._train_metrics,
            }
            torch.save(save_dict, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VisionPredictor":
        """
        モデルを読み込み

        Args:
            path: モデルファイルパス

        Returns:
            VisionPredictorインスタンス
        """
        path = Path(path)

        # 拡張子で判定
        if path.suffix == ".joblib":
            save_dict = joblib.load(path)
            predictor = cls(model_type=save_dict["model_type"])
            predictor.model = save_dict["model"]
            predictor.scaler = save_dict["scaler"]
            predictor._train_metrics = save_dict["train_metrics"]
        else:
            save_dict = torch.load(path, map_location="cpu")
            predictor = cls(model_type=save_dict["model_type"])
            predictor.model = ShallowCNN()
            predictor.model.load_state_dict(save_dict["model_state_dict"])
            predictor.model.to(predictor.device)
            predictor._train_metrics = save_dict["train_metrics"]

        predictor._is_fitted = True
        return predictor


# =============================================================================
# ユーティリティ関数
# =============================================================================

def load_dataset(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    データセットを読み込み

    Args:
        path: vision_dataset.npzのパス

    Returns:
        X, y, match_ids
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    match_ids = data["match_ids"].tolist() if isinstance(data["match_ids"], np.ndarray) else data["match_ids"]

    return X, y, match_ids
