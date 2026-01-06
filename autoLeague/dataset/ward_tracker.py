"""
Ward座標抽出パイプライン - Phase 3

タイムラインデータ（WARD_PLACED/WARD_KILL）とYOLO検出結果をマッチングし、
各wardに一意のIDと座標を付与する。

処理フロー:
1. タイムラインJSONからwardイベントを抽出
2. wards.csv（YOLO検出結果）を読み込み
3. タイムスタンプベースでマッチング
4. 統合結果をCSVで出力
"""

import csv
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# =============================================================================
# 設定
# =============================================================================

# フレーム↔時間変換（デフォルト値、動的計算で上書きされる）
DEFAULT_MS_PER_FRAME = 376  # 旧: 0.047 * 8 * 1000

# マッチング設定
FRAME_TOLERANCE = 10  # マッチング許容フレーム数（約5秒）
IMAGE_SIZE = 512  # ミニマップサイズ（px）

# detection_onlyフィルタリング設定
MIN_CONFIDENCE_DETECTION_ONLY = 0.5  # detection_onlyの最低信頼度（これ未満は除外）

# リバーサイト（スカトルの視界）フィルタリング設定
# これらの座標付近で長時間持続するstealth_wardはリバーサイトとして除外
RIVER_SIGHT_POSITIONS = [
    (362, 332),  # ドラゴン側
    (151, 178),  # バロン側
]
RIVER_SIGHT_RADIUS = 10  # 除外範囲の半径（ピクセル）
RIVER_SIGHT_DURATION_MIN_MS = 75000  # リバーサイト判定の最小持続時間（ミリ秒）
RIVER_SIGHT_DURATION_MAX_MS = 93000  # リバーサイト判定の最大持続時間（ミリ秒）


# =============================================================================
# データ構造
# =============================================================================

@dataclass
class TimelineWardEvent:
    """タイムラインからのwardイベント"""
    timeline_ward_id: int
    event_type: str  # "PLACED" or "KILLED"
    timestamp: int  # ミリ秒
    ward_type: str  # YELLOW_TRINKET, SIGHT_WARD, CONTROL_WARD, UNDEFINED
    participant_id: int  # creatorId or killerId
    team: str  # "blue" or "red"
    frame_expected: int  # 変換後のフレーム番号

    @property
    def is_stealth_ward(self) -> bool:
        # BLUE_TRINKETもYOLOではstealth_wardとして検出される
        return self.ward_type in ("YELLOW_TRINKET", "SIGHT_WARD", "BLUE_TRINKET")

    @property
    def is_control_ward(self) -> bool:
        return self.ward_type == "CONTROL_WARD"


@dataclass
class DetectedWard:
    """YOLO検出結果からのward（クラスタリング済み）"""
    ward_id: int
    class_id: int
    class_name: str
    x: float  # 正規化座標 (0-1)
    y: float
    frame_start: int
    frame_end: int
    detection_count: int
    confidence_avg: float
    matched: bool = False
    timeline_ward_id: Optional[int] = None

    @property
    def team(self) -> str:
        """検出クラスからチームを判定"""
        if "enemy" in self.class_name:
            return "red"
        return "blue"

    @property
    def is_stealth_ward(self) -> bool:
        return "stealth_ward" in self.class_name

    @property
    def is_control_ward(self) -> bool:
        return "control_ward" in self.class_name

    @property
    def x_pixel(self) -> int:
        return int(self.x * IMAGE_SIZE)

    @property
    def y_pixel(self) -> int:
        return int(self.y * IMAGE_SIZE)


@dataclass
class MatchedWard:
    """マッチング後の統合ward情報"""
    ward_id: int
    timeline_ward_id: Optional[int]
    class_name: str
    ward_type: str
    team: str
    x_pixel: int
    y_pixel: int
    x_normalized: float
    y_normalized: float
    frame_start: int
    frame_end: int
    confidence_avg: float
    creator_id: Optional[int]
    timestamp_placed: Optional[int]
    timestamp_killed: Optional[int]
    match_status: str  # "matched", "detection_only", "timeline_only"


# =============================================================================
# タイムラインwardイベント抽出
# =============================================================================

# フィルタリング対象のwardType
# - UNDEFINED: タイムラインで種別が記録されていない（マッチング不可能）
EXCLUDED_WARD_TYPES = {"UNDEFINED"}


def load_ward_events_from_timeline(
    timeline_path: Path,
    ms_per_frame: Optional[float] = DEFAULT_MS_PER_FRAME,
    frame_timestamps: Optional[Dict[int, int]] = None
) -> Tuple[List[TimelineWardEvent], List[TimelineWardEvent], dict]:
    """
    タイムラインJSONからWARD_PLACED/WARD_KILLイベントを抽出

    フィルタリング:
    - creatorId=0 のイベント（無効なparticipantId）
    - wardType='UNDEFINED' のイベント（種別不明）

    Note:
    - BLUE_TRINKETはYOLOモデルでstealth_wardとして検出されるため除外しない

    Args:
        timeline_path: タイムラインJSONのパス
        ms_per_frame: 1フレームあたりのミリ秒（試合ごとに動的計算）
        frame_timestamps: フレームタイムスタンプマップ（存在する場合はこちらを優先）

    Returns:
        (placed_events, killed_events, filter_stats) のタプル
        filter_stats: フィルタリング統計 {"total": int, "filtered": dict, "valid": int}
    """
    with open(timeline_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    placed_events: List[TimelineWardEvent] = []
    killed_events: List[TimelineWardEvent] = []

    # フィルタリング統計
    filter_stats = {
        "total": 0,
        "filtered": {
            "creator_id_zero": 0,
            "undefined": 0
        },
        "valid": 0
    }

    timeline_ward_id = 1

    for frame in data.get("info", {}).get("frames", []):
        for event in frame.get("events", []):
            event_type = event.get("type")

            if event_type == "WARD_PLACED":
                filter_stats["total"] += 1
                creator_id = event.get("creatorId", 0)
                timestamp = event.get("timestamp", 0)
                ward_type = event.get("wardType", "UNDEFINED")

                # フィルタリング: creatorId=0（無効なparticipantId）
                if creator_id == 0:
                    filter_stats["filtered"]["creator_id_zero"] += 1
                    continue

                # フィルタリング: UNDEFINED wardType
                if ward_type == "UNDEFINED":
                    filter_stats["filtered"]["undefined"] += 1
                    continue

                filter_stats["valid"] += 1

                # チーム判定: participantId 1-5 = Blue, 6-10 = Red
                team = "blue" if 1 <= creator_id <= 5 else "red"

                # タイムスタンプからフレーム番号を計算
                if frame_timestamps:
                    frame_expected = timestamp_to_frame_from_map(timestamp, frame_timestamps)
                else:
                    frame_expected = timestamp_to_frame(timestamp, ms_per_frame)

                placed_events.append(TimelineWardEvent(
                    timeline_ward_id=timeline_ward_id,
                    event_type="PLACED",
                    timestamp=timestamp,
                    ward_type=ward_type,
                    participant_id=creator_id,
                    team=team,
                    frame_expected=frame_expected
                ))
                timeline_ward_id += 1

            elif event_type == "WARD_KILL":
                killer_id = event.get("killerId", 0)
                timestamp = event.get("timestamp", 0)
                ward_type = event.get("wardType", "UNDEFINED")

                # KILLイベントはwardTypeがUNDEFINEDでも残す（マッチング用）
                # ただしUNDEFINEDのものはマッチング精度が低い

                # WARD_KILLは破壊したプレイヤーのID
                # wardの所属チームは wardType から推測できない
                team = "unknown"  # 後でマッチング時に判定

                # タイムスタンプからフレーム番号を計算
                if frame_timestamps:
                    frame_expected = timestamp_to_frame_from_map(timestamp, frame_timestamps)
                else:
                    frame_expected = timestamp_to_frame(timestamp, ms_per_frame)

                killed_events.append(TimelineWardEvent(
                    timeline_ward_id=0,  # KILLイベントは別途管理
                    event_type="KILLED",
                    timestamp=timestamp,
                    ward_type=ward_type,
                    participant_id=killer_id,
                    team=team,
                    frame_expected=frame_expected
                ))

    return placed_events, killed_events, filter_stats


def timestamp_to_frame(timestamp_ms: int, ms_per_frame: float = DEFAULT_MS_PER_FRAME) -> int:
    """
    ゲーム内タイムスタンプ（ミリ秒）をフレーム番号に変換

    Args:
        timestamp_ms: ゲーム内タイムスタンプ（ミリ秒）
        ms_per_frame: 1フレームあたりのミリ秒（試合ごとに動的計算）
    """
    return int(timestamp_ms / ms_per_frame)


def frame_to_timestamp(frame: int, ms_per_frame: float = DEFAULT_MS_PER_FRAME) -> int:
    """フレーム番号をゲーム内タイムスタンプ（ミリ秒）に変換"""
    return int(frame * ms_per_frame)


# =============================================================================
# フレームタイムスタンプ読み込み
# =============================================================================

def load_frame_timestamps(frame_timestamps_path: Path) -> Optional[Dict[int, int]]:
    """
    frame_timestamps.csv を読み込み、フレーム番号→ゲーム内時間(ms)のマップを返す

    Args:
        frame_timestamps_path: frame_timestamps.csvのパス

    Returns:
        {frame_number: game_time_ms} の辞書、ファイルがない場合はNone
    """
    if not frame_timestamps_path.exists():
        return None

    frame_to_time: Dict[int, int] = {}
    with open(frame_timestamps_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame_number'])
            time_ms = int(row['game_time_ms'])
            if time_ms >= 0:  # -1 は取得失敗を示す
                frame_to_time[frame] = time_ms

    return frame_to_time if frame_to_time else None


def timestamp_to_frame_from_map(
    timestamp_ms: int,
    frame_timestamps: Dict[int, int]
) -> int:
    """
    フレームタイムスタンプマップを使用してタイムスタンプ→フレーム変換

    Args:
        timestamp_ms: ゲーム内タイムスタンプ（ミリ秒）
        frame_timestamps: {frame_number: game_time_ms} の辞書

    Returns:
        最も近いゲーム内時間を持つフレーム番号
    """
    best_frame = 0
    best_diff = float('inf')

    for frame, time_ms in frame_timestamps.items():
        diff = abs(time_ms - timestamp_ms)
        if diff < best_diff:
            best_diff = diff
            best_frame = frame

    return best_frame


def frame_to_timestamp_from_map(
    frame: int,
    frame_timestamps: Dict[int, int]
) -> int:
    """
    フレームタイムスタンプマップを使用してフレーム→タイムスタンプ変換

    Args:
        frame: フレーム番号
        frame_timestamps: {frame_number: game_time_ms} の辞書

    Returns:
        ゲーム内タイムスタンプ（ミリ秒）
    """
    if frame in frame_timestamps:
        return frame_timestamps[frame]

    # 該当フレームがない場合、最も近いフレームを探す
    frames = sorted(frame_timestamps.keys())
    if not frames:
        return 0

    if frame <= frames[0]:
        return frame_timestamps[frames[0]]
    if frame >= frames[-1]:
        return frame_timestamps[frames[-1]]

    # 二分探索で近いフレームを探す
    for i, f in enumerate(frames):
        if f > frame:
            # frames[i-1] と frames[i] の間で補間
            prev_frame = frames[i - 1]
            next_frame = f
            prev_time = frame_timestamps[prev_frame]
            next_time = frame_timestamps[next_frame]
            # 線形補間
            ratio = (frame - prev_frame) / (next_frame - prev_frame)
            return int(prev_time + ratio * (next_time - prev_time))

    return frame_timestamps[frames[-1]]


# =============================================================================
# YOLO検出結果読み込み
# =============================================================================

def load_detected_wards(wards_csv_path: Path) -> List[DetectedWard]:
    """wards.csv（クラスタリング済み検出結果）を読み込み"""
    wards: List[DetectedWard] = []

    with open(wards_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wards.append(DetectedWard(
                ward_id=int(row['ward_id']),
                class_id=int(row['class_id']),
                class_name=row['class_name'],
                x=float(row['x']),
                y=float(row['y']),
                frame_start=int(row['frame_start']),
                frame_end=int(row['frame_end']),
                detection_count=int(row['detection_count']),
                confidence_avg=float(row['confidence_avg'])
            ))

    return wards


# =============================================================================
# マッチングロジック
# =============================================================================

def match_wards(
    placed_events: List[TimelineWardEvent],
    killed_events: List[TimelineWardEvent],
    detected_wards: List[DetectedWard],
    frame_tolerance: int = FRAME_TOLERANCE,
    ms_per_frame: Optional[float] = DEFAULT_MS_PER_FRAME,
    frame_timestamps: Optional[Dict[int, int]] = None
) -> List[MatchedWard]:
    """
    タイムラインイベントと検出結果をマッチング

    アルゴリズム:
    1. タイムラインのward設置イベントをタイムスタンプ順にソート
    2. 各設置イベントに対して:
       - 設置タイムスタンプをフレーム番号に変換（frame_expected）
       - frame_expected以降で新規出現したwardクラスタを検索
       - 同一チーム・wardTypeの中で最も信頼度が高いものを採用
    3. マッチングしなかった検出結果はdetection_onlyとして記録
    4. マッチングしなかったタイムラインイベントはtimeline_onlyとして記録

    Args:
        ms_per_frame: 1フレームあたりのミリ秒（試合ごとに動的計算）
        frame_timestamps: フレームタイムスタンプマップ（存在する場合はこちらを優先）
    """
    matched_wards: List[MatchedWard] = []
    result_ward_id = 1

    # タイムスタンプ順にソート
    sorted_placed = sorted(placed_events, key=lambda e: e.timestamp)

    # KILLイベントをタイムスタンプでインデックス化（後で使用）
    kill_by_time: Dict[int, TimelineWardEvent] = {}
    for ke in killed_events:
        kill_by_time[ke.timestamp] = ke

    # 1. 各設置イベントに対してマッチング
    for event in sorted_placed:
        # 候補となるward検出を検索
        candidates = find_matching_candidates(event, detected_wards, frame_tolerance)

        if candidates:
            # 最も信頼度が高いwardを選択
            best = max(candidates, key=lambda w: w.confidence_avg)
            best.matched = True
            best.timeline_ward_id = event.timeline_ward_id

            # KILL時刻を検索（同じwardTypeで最も近いKILLイベント）
            kill_timestamp = find_kill_timestamp(
                event, killed_events, best.frame_end, ms_per_frame, frame_timestamps
            )

            matched_wards.append(MatchedWard(
                ward_id=result_ward_id,
                timeline_ward_id=event.timeline_ward_id,
                class_name=best.class_name,
                ward_type=event.ward_type,
                team=event.team,
                x_pixel=best.x_pixel,
                y_pixel=best.y_pixel,
                x_normalized=best.x,
                y_normalized=best.y,
                frame_start=best.frame_start,
                frame_end=best.frame_end,
                confidence_avg=best.confidence_avg,
                creator_id=event.participant_id,
                timestamp_placed=event.timestamp,
                timestamp_killed=kill_timestamp,
                match_status="matched"
            ))
            result_ward_id += 1
        else:
            # 検出されなかったwardはtimeline_onlyとして記録
            matched_wards.append(MatchedWard(
                ward_id=result_ward_id,
                timeline_ward_id=event.timeline_ward_id,
                class_name=get_expected_class_name(event),
                ward_type=event.ward_type,
                team=event.team,
                x_pixel=-1,  # 座標不明
                y_pixel=-1,
                x_normalized=-1.0,
                y_normalized=-1.0,
                frame_start=event.frame_expected,
                frame_end=-1,
                confidence_avg=0.0,
                creator_id=event.participant_id,
                timestamp_placed=event.timestamp,
                timestamp_killed=None,
                match_status="timeline_only"
            ))
            result_ward_id += 1

    # 2. マッチングしなかった検出結果をdetection_onlyとして追加
    # 低信頼度のwardはフィルタリング（誤検出の可能性が高い）
    # ※リバーサイトは事前にfilter_river_sightsで除外済み
    for ward in detected_wards:
        if not ward.matched:
            # 低信頼度フィルタリング
            if ward.confidence_avg < MIN_CONFIDENCE_DETECTION_ONLY:
                continue

            # フレーム→タイムスタンプ変換
            if frame_timestamps:
                ts_placed = frame_to_timestamp_from_map(ward.frame_start, frame_timestamps)
            else:
                ts_placed = frame_to_timestamp(ward.frame_start, ms_per_frame)

            matched_wards.append(MatchedWard(
                ward_id=result_ward_id,
                timeline_ward_id=None,
                class_name=ward.class_name,
                ward_type=get_ward_type_from_class(ward.class_name),
                team=ward.team,
                x_pixel=ward.x_pixel,
                y_pixel=ward.y_pixel,
                x_normalized=ward.x,
                y_normalized=ward.y,
                frame_start=ward.frame_start,
                frame_end=ward.frame_end,
                confidence_avg=ward.confidence_avg,
                creator_id=None,
                timestamp_placed=ts_placed,
                timestamp_killed=None,
                match_status="detection_only"
            ))
            result_ward_id += 1

    return matched_wards


def find_matching_candidates(
    event: TimelineWardEvent,
    detected_wards: List[DetectedWard],
    frame_tolerance: int
) -> List[DetectedWard]:
    """
    タイムラインイベントにマッチする検出候補を検索

    条件:
    1. 未マッチングのward
    2. フレーム開始がevent.frame_expected以降（±toleranceの許容範囲）
    3. 同一チーム
    4. 同一wardType（stealth/control）
    """
    candidates = []

    for ward in detected_wards:
        if ward.matched:
            continue

        # フレーム範囲チェック
        frame_diff = ward.frame_start - event.frame_expected
        if frame_diff < -frame_tolerance or frame_diff > frame_tolerance * 3:
            continue

        # チームチェック
        if ward.team != event.team:
            continue

        # wardTypeチェック
        if event.is_stealth_ward and not ward.is_stealth_ward:
            continue
        if event.is_control_ward and not ward.is_control_ward:
            continue

        candidates.append(ward)

    return candidates


def find_kill_timestamp(
    placed_event: TimelineWardEvent,
    killed_events: List[TimelineWardEvent],
    detection_frame_end: int,
    ms_per_frame: Optional[float] = DEFAULT_MS_PER_FRAME,
    frame_timestamps: Optional[Dict[int, int]] = None
) -> Optional[int]:
    """
    wardの破壊タイムスタンプを検索

    設置時刻以降で、検出終了フレーム付近のKILLイベントを検索
    """
    if frame_timestamps:
        detection_end_ts = frame_to_timestamp_from_map(detection_frame_end, frame_timestamps)
    else:
        detection_end_ts = frame_to_timestamp(detection_frame_end, ms_per_frame)

    # 同じwardTypeで、設置後に発生したKILLイベントを検索
    candidates = [
        ke for ke in killed_events
        if ke.timestamp > placed_event.timestamp
        and ke.ward_type == placed_event.ward_type
    ]

    if not candidates:
        return None

    # 検出終了時刻に最も近いKILLイベントを選択
    best = min(candidates, key=lambda ke: abs(ke.timestamp - detection_end_ts))

    # 許容範囲内であれば採用（30秒以内）
    if abs(best.timestamp - detection_end_ts) < 30000:
        return best.timestamp

    return None


def get_expected_class_name(event: TimelineWardEvent) -> str:
    """タイムラインイベントから期待されるクラス名を生成"""
    if event.is_control_ward:
        return "control_ward" if event.team == "blue" else "control_ward_enemy"
    else:
        return "stealth_ward" if event.team == "blue" else "stealth_ward_enemy"


def get_ward_type_from_class(class_name: str) -> str:
    """検出クラス名からwardTypeを推測"""
    if "control_ward" in class_name:
        return "CONTROL_WARD"
    else:
        return "SIGHT_WARD"  # デフォルトはSIGHT_WARD


def is_river_sight_position(x_pixel: int, y_pixel: int) -> bool:
    """
    リバーサイト位置かどうかを判定（座標のみ）

    Args:
        x_pixel: X座標（ピクセル）
        y_pixel: Y座標（ピクセル）

    Returns:
        リバーサイト位置の場合True
    """
    for rx, ry in RIVER_SIGHT_POSITIONS:
        dist = np.sqrt((x_pixel - rx) ** 2 + (y_pixel - ry) ** 2)
        if dist <= RIVER_SIGHT_RADIUS:
            return True
    return False


def filter_river_sights(
    detected_wards: List[DetectedWard],
    frame_timestamps: Optional[Dict[int, int]] = None,
    ms_per_frame: Optional[float] = DEFAULT_MS_PER_FRAME
) -> Tuple[List[DetectedWard], int]:
    """
    リバーサイト（スカトルの視界）を検出結果から除外

    判定条件:
    1. stealth_wardである
    2. リバーサイト位置（固定座標付近）にある
    3. 持続時間が75~93秒（リバーサイトの90秒に該当）

    Args:
        detected_wards: YOLO検出結果リスト
        frame_timestamps: フレームタイムスタンプマップ
        ms_per_frame: 1フレームあたりのミリ秒

    Returns:
        (フィルタリング後のリスト, 除外されたward数)
    """
    filtered = []
    removed_count = 0

    for ward in detected_wards:
        # stealth_wardのみ対象
        if not ward.is_stealth_ward:
            filtered.append(ward)
            continue

        # リバーサイト位置チェック
        if not is_river_sight_position(ward.x_pixel, ward.y_pixel):
            filtered.append(ward)
            continue

        # 持続時間を計算
        if frame_timestamps:
            start_ts = frame_to_timestamp_from_map(ward.frame_start, frame_timestamps)
            end_ts = frame_to_timestamp_from_map(ward.frame_end, frame_timestamps)
            duration_ms = end_ts - start_ts
        else:
            duration_frames = ward.frame_end - ward.frame_start
            duration_ms = duration_frames * ms_per_frame

        # 持続時間チェック（75~93秒ならリバーサイト）
        if RIVER_SIGHT_DURATION_MIN_MS <= duration_ms < RIVER_SIGHT_DURATION_MAX_MS:
            removed_count += 1
            continue  # リバーサイトとして除外

        filtered.append(ward)

    return filtered, removed_count


# =============================================================================
# ハンガリアン法マッチング
# =============================================================================

def match_wards_hungarian(
    placed_events: List[TimelineWardEvent],
    killed_events: List[TimelineWardEvent],
    detected_wards: List[DetectedWard],
    frame_tolerance: int = FRAME_TOLERANCE,
    ms_per_frame: Optional[float] = DEFAULT_MS_PER_FRAME,
    frame_timestamps: Optional[Dict[int, int]] = None,
    ignore_team: bool = False
) -> List[MatchedWard]:
    """
    ハンガリアン法による全体最適マッチング

    貪欲法と異なり、全体のコスト（フレーム差の合計）が最小になる
    割り当てを計算する。これにより、マッチング競合を解消できる。

    Args:
        placed_events: タイムラインのward設置イベントリスト
        killed_events: タイムラインのward破壊イベントリスト
        detected_wards: YOLO検出結果リスト
        frame_tolerance: マッチング許容フレーム数
        ms_per_frame: 1フレームあたりのミリ秒
        frame_timestamps: フレームタイムスタンプマップ
        ignore_team: Trueの場合、チーム不一致でもマッチング候補とする
    """
    matched_wards: List[MatchedWard] = []
    result_ward_id = 1

    # ward_type別にグループ化してマッチング
    # (stealth wardとcontrol wardを別々に処理)
    for ward_type_group in ["stealth", "control"]:
        # タイムラインイベントをフィルタ
        if ward_type_group == "stealth":
            events = [e for e in placed_events if e.is_stealth_ward]
        else:
            events = [e for e in placed_events if e.is_control_ward]

        # 検出wardをフィルタ
        if ward_type_group == "stealth":
            detections = [w for w in detected_wards if w.is_stealth_ward and not w.matched]
        else:
            detections = [w for w in detected_wards if w.is_control_ward and not w.matched]

        if not events or not detections:
            # マッチング対象がない場合はスキップ
            for event in events:
                matched_wards.append(MatchedWard(
                    ward_id=result_ward_id,
                    timeline_ward_id=event.timeline_ward_id,
                    class_name=get_expected_class_name(event),
                    ward_type=event.ward_type,
                    team=event.team,
                    x_pixel=-1,
                    y_pixel=-1,
                    x_normalized=-1.0,
                    y_normalized=-1.0,
                    frame_start=event.frame_expected,
                    frame_end=-1,
                    confidence_avg=0.0,
                    creator_id=event.participant_id,
                    timestamp_placed=event.timestamp,
                    timestamp_killed=None,
                    match_status="timeline_only"
                ))
                result_ward_id += 1
            continue

        # コスト行列を作成
        n_events = len(events)
        n_detections = len(detections)
        INF_COST = 1e9  # マッチング不可能なペアのコスト

        cost_matrix = np.full((n_events, n_detections), INF_COST)

        for i, event in enumerate(events):
            for j, ward in enumerate(detections):
                # フレーム差
                frame_diff = ward.frame_start - event.frame_expected

                # tolerance範囲外は対象外
                if frame_diff < -frame_tolerance or frame_diff > frame_tolerance * 3:
                    continue

                # チームチェック（ignore_team=Falseの場合のみ）
                if not ignore_team and ward.team != event.team:
                    continue

                # コスト = フレーム差の絶対値（小さいほど良い）
                # confidenceも考慮: 高いほどコストを下げる
                cost = abs(frame_diff) - ward.confidence_avg * 10
                cost_matrix[i, j] = cost

        # ハンガリアン法で最適割り当てを計算
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # マッチング結果を処理
        matched_event_indices = set()
        matched_detection_indices = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < INF_COST:
                # 有効なマッチング
                event = events[i]
                ward = detections[j]

                ward.matched = True
                ward.timeline_ward_id = event.timeline_ward_id
                matched_event_indices.add(i)
                matched_detection_indices.add(j)

                # KILL時刻を検索
                kill_timestamp = find_kill_timestamp(
                    event, killed_events, ward.frame_end, ms_per_frame, frame_timestamps
                )

                matched_wards.append(MatchedWard(
                    ward_id=result_ward_id,
                    timeline_ward_id=event.timeline_ward_id,
                    class_name=ward.class_name,
                    ward_type=event.ward_type,
                    team=event.team,  # タイムラインのチーム情報を優先
                    x_pixel=ward.x_pixel,
                    y_pixel=ward.y_pixel,
                    x_normalized=ward.x,
                    y_normalized=ward.y,
                    frame_start=ward.frame_start,
                    frame_end=ward.frame_end,
                    confidence_avg=ward.confidence_avg,
                    creator_id=event.participant_id,
                    timestamp_placed=event.timestamp,
                    timestamp_killed=kill_timestamp,
                    match_status="matched"
                ))
                result_ward_id += 1

        # マッチングしなかったタイムラインイベントをtimeline_onlyとして追加
        for i, event in enumerate(events):
            if i not in matched_event_indices:
                matched_wards.append(MatchedWard(
                    ward_id=result_ward_id,
                    timeline_ward_id=event.timeline_ward_id,
                    class_name=get_expected_class_name(event),
                    ward_type=event.ward_type,
                    team=event.team,
                    x_pixel=-1,
                    y_pixel=-1,
                    x_normalized=-1.0,
                    y_normalized=-1.0,
                    frame_start=event.frame_expected,
                    frame_end=-1,
                    confidence_avg=0.0,
                    creator_id=event.participant_id,
                    timestamp_placed=event.timestamp,
                    timestamp_killed=None,
                    match_status="timeline_only"
                ))
                result_ward_id += 1

    # マッチングしなかった検出結果をdetection_onlyとして追加
    # 低信頼度のwardはフィルタリング（誤検出の可能性が高い）
    # ※リバーサイトは事前にfilter_river_sightsで除外済み
    for ward in detected_wards:
        if not ward.matched:
            # 低信頼度フィルタリング
            if ward.confidence_avg < MIN_CONFIDENCE_DETECTION_ONLY:
                continue

            if frame_timestamps:
                ts_placed = frame_to_timestamp_from_map(ward.frame_start, frame_timestamps)
            else:
                ts_placed = frame_to_timestamp(ward.frame_start, ms_per_frame)

            matched_wards.append(MatchedWard(
                ward_id=result_ward_id,
                timeline_ward_id=None,
                class_name=ward.class_name,
                ward_type=get_ward_type_from_class(ward.class_name),
                team=ward.team,
                x_pixel=ward.x_pixel,
                y_pixel=ward.y_pixel,
                x_normalized=ward.x,
                y_normalized=ward.y,
                frame_start=ward.frame_start,
                frame_end=ward.frame_end,
                confidence_avg=ward.confidence_avg,
                creator_id=None,
                timestamp_placed=ts_placed,
                timestamp_killed=None,
                match_status="detection_only"
            ))
            result_ward_id += 1

    return matched_wards


# =============================================================================
# 出力
# =============================================================================

def save_matched_wards(matched_wards: List[MatchedWard], output_path: Path):
    """マッチング結果をCSVに保存"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'ward_id', 'timeline_ward_id', 'class_name', 'ward_type', 'team',
            'x_pixel', 'y_pixel', 'x_normalized', 'y_normalized',
            'frame_start', 'frame_end', 'confidence_avg',
            'creator_id', 'timestamp_placed', 'timestamp_killed', 'match_status'
        ])

        for w in matched_wards:
            writer.writerow([
                w.ward_id,
                w.timeline_ward_id if w.timeline_ward_id else '',
                w.class_name,
                w.ward_type,
                w.team,
                w.x_pixel if w.x_pixel >= 0 else '',
                w.y_pixel if w.y_pixel >= 0 else '',
                f"{w.x_normalized:.6f}" if w.x_normalized >= 0 else '',
                f"{w.y_normalized:.6f}" if w.y_normalized >= 0 else '',
                w.frame_start,
                w.frame_end if w.frame_end >= 0 else '',
                f"{w.confidence_avg:.4f}" if w.confidence_avg > 0 else '',
                w.creator_id if w.creator_id else '',
                w.timestamp_placed if w.timestamp_placed else '',
                w.timestamp_killed if w.timestamp_killed else '',
                w.match_status
            ])

    print(f"マッチング結果を保存: {output_path}")


# =============================================================================
# 統計表示
# =============================================================================

def print_matching_statistics(matched_wards: List[MatchedWard], placed_count: int, detected_count: int):
    """マッチング結果の統計を表示"""
    print("\n=== マッチング結果 ===")
    print(f"タイムラインward設置イベント数: {placed_count}")
    print(f"YOLO検出ward数: {detected_count}")
    print(f"出力ward数: {len(matched_wards)}")

    # ステータス別集計
    status_counts = {}
    for w in matched_wards:
        status_counts[w.match_status] = status_counts.get(w.match_status, 0) + 1

    print("\nステータス別:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # マッチング率
    matched_count = status_counts.get("matched", 0)
    if placed_count > 0:
        match_rate = matched_count / placed_count * 100
        print(f"\nマッチング率: {match_rate:.1f}% ({matched_count}/{placed_count})")

    # チーム別集計
    team_counts = {"blue": 0, "red": 0}
    for w in matched_wards:
        if w.team in team_counts:
            team_counts[w.team] += 1

    print("\nチーム別ward数:")
    for team, count in team_counts.items():
        print(f"  {team}: {count}")


# =============================================================================
# メイン処理
# =============================================================================

class WardTracker:
    """Ward座標抽出パイプラインのメインクラス"""

    def __init__(
        self,
        timeline_dir: Path = Path("data/timeline"),
        dataset_dir: Path = Path(r"C:\dataset_20260105"),
        frame_tolerance: int = FRAME_TOLERANCE,
        use_hungarian: bool = False,
        ignore_team: bool = False
    ):
        self.timeline_dir = timeline_dir
        self.dataset_dir = dataset_dir
        self.frame_tolerance = frame_tolerance
        self.use_hungarian = use_hungarian
        self.ignore_team = ignore_team

    def _calculate_ms_per_frame(self, timeline_path: Path, frame_dir: Path) -> float:
        """
        試合ごとに動的にフレーム変換係数を計算

        Args:
            timeline_path: タイムラインJSONのパス
            frame_dir: キャプチャフレームフォルダのパス

        Returns:
            1フレームあたりのミリ秒
        """
        # タイムラインからゲーム時間を取得
        with open(timeline_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        game_duration_ms = data["info"]["frames"][-1]["timestamp"]

        # フレーム数を取得
        if frame_dir.exists():
            total_frames = len(list(frame_dir.glob("*.png")))
        else:
            # フレームフォルダがない場合はデフォルト値を使用
            print(f"警告: フレームフォルダが見つかりません: {frame_dir}")
            return DEFAULT_MS_PER_FRAME

        if total_frames == 0:
            print(f"警告: フレームが見つかりません: {frame_dir}")
            return DEFAULT_MS_PER_FRAME

        # 動的に計算
        ms_per_frame = game_duration_ms / total_frames
        return ms_per_frame

    def process_match(self, match_id: str) -> List[MatchedWard]:
        """
        1試合を処理

        Args:
            match_id: 試合ID（例: "JP1-555621265" または "555621265"）

        Returns:
            マッチング結果のリスト
        """
        # match_idの正規化
        if match_id.startswith("JP1-"):
            match_id_num = match_id.replace("JP1-", "")
            match_id_full = match_id
        else:
            match_id_num = match_id
            match_id_full = f"JP1-{match_id}"

        print(f"\n{'='*60}")
        print(f"処理開始: {match_id_full}")
        print(f"{'='*60}")

        # パス設定
        timeline_path = self.timeline_dir / f"JP1_{match_id_num}.json"
        match_dir = self.dataset_dir / match_id_full
        wards_csv_path = match_dir / "wards.csv"
        output_path = match_dir / "wards_matched.csv"
        frame_dir = match_dir / "0"  # キャプチャフレームフォルダ
        frame_timestamps_path = match_dir / "frame_timestamps.csv"

        # ファイル存在チェック
        if not timeline_path.exists():
            print(f"エラー: タイムラインファイルが見つかりません: {timeline_path}")
            return []

        if not wards_csv_path.exists():
            print(f"エラー: 検出結果ファイルが見つかりません: {wards_csv_path}")
            print("先にbatch_inference_wards.pyを実行してください")
            return []

        # 0. フレームタイムスタンプを読み込み（存在する場合）
        frame_timestamps = load_frame_timestamps(frame_timestamps_path)
        if frame_timestamps:
            print(f"\n0. フレームタイムスタンプ使用: {frame_timestamps_path}")
            print(f"   フレーム数: {len(frame_timestamps)}")
            ms_per_frame = None  # フレームタイムスタンプ使用時は不要
        else:
            # フレームタイムスタンプがない場合は動的計算
            ms_per_frame = self._calculate_ms_per_frame(timeline_path, frame_dir)
            print(f"\n0. フレーム変換係数（動的計算）: {ms_per_frame:.2f} ms/frame")

        # 1. タイムラインからwardイベントを抽出
        print(f"\n1. タイムラインデータ読み込み: {timeline_path}")
        placed_events, killed_events, filter_stats = load_ward_events_from_timeline(
            timeline_path, ms_per_frame, frame_timestamps
        )
        print(f"   WARD_PLACED(全体): {filter_stats['total']}件")
        print(f"   フィルタリング:")
        print(f"     - creatorId=0: {filter_stats['filtered']['creator_id_zero']}件（除外）")
        print(f"     - UNDEFINED: {filter_stats['filtered']['undefined']}件（除外）")
        print(f"   有効イベント: {filter_stats['valid']}件")
        print(f"   WARD_KILL: {len(killed_events)}件")

        # 2. YOLO検出結果を読み込み
        print(f"\n2. 検出結果読み込み: {wards_csv_path}")
        detected_wards = load_detected_wards(wards_csv_path)
        print(f"   検出ward数: {len(detected_wards)}")

        # 2.5. リバーサイト（スカトルの視界）をフィルタリング
        detected_wards, river_sight_count = filter_river_sights(
            detected_wards, frame_timestamps, ms_per_frame
        )
        if river_sight_count > 0:
            print(f"   リバーサイト除外: {river_sight_count}件")
            print(f"   フィルタリング後: {len(detected_wards)}件")

        # 3. マッチング
        if self.use_hungarian:
            print(f"\n3. マッチング実行 [ハンガリアン法] (tolerance={self.frame_tolerance}フレーム, ignore_team={self.ignore_team})")
            matched_wards = match_wards_hungarian(
                placed_events, killed_events, detected_wards,
                self.frame_tolerance, ms_per_frame, frame_timestamps,
                ignore_team=self.ignore_team
            )
        else:
            print(f"\n3. マッチング実行 [貪欲法] (tolerance={self.frame_tolerance}フレーム)")
            matched_wards = match_wards(
                placed_events, killed_events, detected_wards,
                self.frame_tolerance, ms_per_frame, frame_timestamps
            )

        # 4. 結果保存
        print(f"\n4. 結果保存: {output_path}")
        save_matched_wards(matched_wards, output_path)

        # 5. 統計表示
        print_matching_statistics(matched_wards, filter_stats['valid'], len(detected_wards))

        return matched_wards

    def process_all(self) -> Dict[str, List[MatchedWard]]:
        """全試合を処理"""
        results = {}

        # データセット内の全試合ディレクトリを検索
        match_dirs = sorted(self.dataset_dir.glob("JP1-*"))
        print(f"全{len(match_dirs)}試合を処理します")

        for match_dir in match_dirs:
            match_id = match_dir.name
            try:
                matched = self.process_match(match_id)
                results[match_id] = matched
            except Exception as e:
                print(f"エラー [{match_id}]: {e}")

        print(f"\n処理完了: {len(results)}試合")
        return results


# =============================================================================
# スタンドアロン実行
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ward座標抽出パイプライン")
    parser.add_argument("--match", type=str, help="処理する試合ID（例: JP1-555621265）")
    parser.add_argument("--all", action="store_true", help="全試合を処理")
    parser.add_argument("--timeline-dir", type=str, default="data/timeline",
                        help="タイムラインデータディレクトリ")
    parser.add_argument("--dataset", type=str, default=r"C:\dataset_20260105",
                        help="データセットディレクトリ")
    parser.add_argument("--tolerance", type=int, default=FRAME_TOLERANCE,
                        help="マッチング許容フレーム数")
    parser.add_argument("--hungarian", action="store_true",
                        help="ハンガリアン法（全体最適）を使用")
    parser.add_argument("--ignore-team", action="store_true",
                        help="チーム不一致でもマッチング候補とする（ハンガリアン法使用時のみ有効）")
    args = parser.parse_args()

    tracker = WardTracker(
        timeline_dir=Path(args.timeline_dir),
        dataset_dir=Path(args.dataset),
        frame_tolerance=args.tolerance,
        use_hungarian=args.hungarian,
        ignore_team=args.ignore_team
    )

    if args.match:
        tracker.process_match(args.match)
    elif args.all:
        tracker.process_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
