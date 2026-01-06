"""
Ward マッチング結果HTMLレポート生成（フレーム画像付き）

使用方法:
    python scripts/generate_ward_report.py --match JP1-555621265
    python scripts/generate_ward_report.py --all
"""

import argparse
import csv
import base64
from pathlib import Path
from typing import List, Dict
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

DATASET_DIR = Path(r"C:\dataset_20260105")
OUTPUT_DIR = Path("reports")
IMAGE_SIZE = 512

# クラス別の色（BGR）
CLASS_COLORS = {
    "stealth_ward": (0, 255, 0),           # 緑
    "stealth_ward_enemy": (0, 0, 255),     # 赤
    "control_ward": (255, 255, 0),         # シアン
    "control_ward_enemy": (0, 165, 255),   # オレンジ
}

# ステータス別の色（BGR）
STATUS_COLORS = {
    "matched": (0, 255, 0),        # 緑
    "timeline_only": (0, 0, 255),  # 赤
    "detection_only": (255, 0, 0), # 青
}


def load_wards_matched(csv_path: Path) -> List[Dict]:
    """wards_matched.csvを読み込み"""
    wards = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wards.append({
                'ward_id': int(row['ward_id']),
                'timeline_ward_id': int(row['timeline_ward_id']) if row['timeline_ward_id'] else None,
                'class_name': row['class_name'],
                'ward_type': row['ward_type'],
                'team': row['team'],
                'x_pixel': int(row['x_pixel']) if row['x_pixel'] else -1,
                'y_pixel': int(row['y_pixel']) if row['y_pixel'] else -1,
                'frame_start': int(row['frame_start']) if row['frame_start'] else 0,
                'frame_end': int(row['frame_end']) if row['frame_end'] else 0,
                'confidence_avg': float(row['confidence_avg']) if row['confidence_avg'] else 0,
                'creator_id': row['creator_id'],
                'timestamp_placed': int(row['timestamp_placed']) if row['timestamp_placed'] else 0,
                'match_status': row['match_status']
            })
    return wards


def draw_ward_box(img: np.ndarray, ward: Dict) -> np.ndarray:
    """画像上にwardのバウンディングボックスを描画"""
    img_copy = img.copy()

    x = ward['x_pixel']
    y = ward['y_pixel']

    if x < 0 or y < 0:
        return img_copy

    # wardのサイズ
    box_size = 24
    x1 = max(0, x - box_size // 2)
    y1 = max(0, y - box_size // 2)
    x2 = min(IMAGE_SIZE, x + box_size // 2)
    y2 = min(IMAGE_SIZE, y + box_size // 2)

    # ステータス別の色
    color = STATUS_COLORS.get(ward['match_status'], (255, 255, 255))

    # バウンディングボックス
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

    # ラベル
    label = f"#{ward['ward_id']}"
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # ラベル背景
    cv2.rectangle(img_copy, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(img_copy, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness)

    return img_copy


def generate_ward_image(match_dir: Path, ward: Dict) -> str:
    """wardの開始/中間/終了フレームを並べた画像を生成してBase64で返す"""
    frame_dir = match_dir / "0"

    frame_start = ward['frame_start']
    frame_end = ward['frame_end'] if ward['frame_end'] > 0 else frame_start

    # 3つのフレームを選択
    frames_to_show = [
        frame_start,
        (frame_start + frame_end) // 2,
        frame_end
    ]

    images = []
    for frame_num in frames_to_show:
        frame_path = frame_dir / f"{frame_num}.png"
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            img = draw_ward_box(img, ward)
            # フレーム番号を追加
            cv2.putText(img, f"Frame: {frame_num}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            images.append(img)
        else:
            # フレームがない場合は空の画像
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            cv2.putText(img, f"Frame {frame_num} not found", (50, IMAGE_SIZE//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            images.append(img)

    if not images:
        return ""

    # 横に連結
    combined = np.hstack(images)

    # リサイズ（幅を800pxに）
    scale = 800 / combined.shape[1]
    new_size = (800, int(combined.shape[0] * scale))
    combined = cv2.resize(combined, new_size)

    # Base64エンコード
    _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def generate_html_report(match_id: str, wards: List[Dict], match_dir: Path) -> str:
    """HTMLレポートを生成"""

    # 統計計算
    matched = [w for w in wards if w['match_status'] == 'matched']
    timeline_only = [w for w in wards if w['match_status'] == 'timeline_only']
    detection_only = [w for w in wards if w['match_status'] == 'detection_only']
    total_timeline = len(matched) + len(timeline_only)
    match_rate = len(matched) / total_timeline * 100 if total_timeline > 0 else 0

    # wardの画像を生成
    print(f"  wardの画像を生成中...")
    ward_images = {}
    for ward in wards:
        if ward['x_pixel'] >= 0:  # 座標があるwardのみ
            ward_images[ward['ward_id']] = generate_ward_image(match_dir, ward)

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>Ward Report - {match_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; margin-bottom: 20px; }}
        h2 {{ color: #00d4ff; margin: 20px 0 15px; font-size: 1.3em; }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card.matched {{ border-left: 4px solid #2ecc71; }}
        .stat-card.timeline-only {{ border-left: 4px solid #e74c3c; }}
        .stat-card.detection-only {{ border-left: 4px solid #3498db; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; }}
        .stat-label {{ color: #888; font-size: 0.85em; margin-top: 5px; }}

        .section {{ margin-bottom: 40px; }}
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .section-header h2 {{ margin: 0; }}
        .toggle-btn {{
            background: #0f3460;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
        }}
        .toggle-btn:hover {{ background: #1a4a7a; }}

        .ward-grid {{ display: flex; flex-direction: column; gap: 20px; }}
        .ward-card {{
            background: #16213e;
            border-radius: 10px;
            overflow: hidden;
        }}
        .ward-card.matched {{ border-left: 4px solid #2ecc71; }}
        .ward-card.timeline-only {{ border-left: 4px solid #e74c3c; }}
        .ward-card.detection-only {{ border-left: 4px solid #3498db; }}

        .ward-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            background: #0f3460;
            cursor: pointer;
        }}
        .ward-header:hover {{ background: #1a4a7a; }}
        .ward-title {{ font-weight: bold; }}
        .ward-meta {{ color: #888; font-size: 0.9em; }}

        .ward-content {{
            padding: 15px;
            display: none;
        }}
        .ward-content.show {{ display: block; }}
        .ward-image {{
            width: 100%;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .ward-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        .ward-detail {{
            background: #0f3460;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        .ward-detail-label {{ color: #888; font-size: 0.8em; }}

        .status-matched {{ color: #2ecc71; }}
        .status-timeline_only {{ color: #e74c3c; }}
        .status-detection_only {{ color: #3498db; }}
        .team-blue {{ color: #3498db; }}
        .team-red {{ color: #e74c3c; }}

        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.9em; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .legend-dot.matched {{ background: #2ecc71; }}
        .legend-dot.timeline-only {{ background: #e74c3c; }}
        .legend-dot.detection-only {{ background: #3498db; }}

        .back-link {{ color: #00d4ff; text-decoration: none; margin-bottom: 20px; display: inline-block; }}
        .back-link:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-link">&larr; Back to Index</a>
        <h1>Ward Matching Report - {match_id}</h1>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{match_rate:.1f}%</div>
                <div class="stat-label">Match Rate</div>
            </div>
            <div class="stat-card matched">
                <div class="stat-value">{len(matched)}</div>
                <div class="stat-label">Matched</div>
            </div>
            <div class="stat-card timeline-only">
                <div class="stat-value">{len(timeline_only)}</div>
                <div class="stat-label">Timeline Only</div>
            </div>
            <div class="stat-card detection-only">
                <div class="stat-value">{len(detection_only)}</div>
                <div class="stat-label">Detection Only</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_timeline}</div>
                <div class="stat-label">Timeline Total</div>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item"><div class="legend-dot matched"></div>Matched (Timeline + YOLO)</div>
            <div class="legend-item"><div class="legend-dot timeline-only"></div>Timeline Only (YOLO failed)</div>
            <div class="legend-item"><div class="legend-dot detection-only"></div>Detection Only (No timeline)</div>
        </div>
'''

    # セクション別にward表示
    sections = [
        ("Matched Wards", matched, "matched"),
        ("Timeline Only (Detection Failed)", timeline_only, "timeline-only"),
        ("Detection Only", detection_only, "detection-only"),
    ]

    for section_title, section_wards, section_class in sections:
        if not section_wards:
            continue

        html += f'''
        <div class="section">
            <div class="section-header">
                <h2>{section_title} ({len(section_wards)})</h2>
                <button class="toggle-btn" onclick="toggleSection(this)">Expand All</button>
            </div>
            <div class="ward-grid">
'''

        for ward in sorted(section_wards, key=lambda w: w['ward_id']):
            img_data = ward_images.get(ward['ward_id'], "")
            has_image = bool(img_data)

            duration = ward['frame_end'] - ward['frame_start'] if ward['frame_end'] > 0 else 0

            html += f'''
                <div class="ward-card {section_class}">
                    <div class="ward-header" onclick="toggleWard(this)">
                        <span class="ward-title">
                            Ward #{ward['ward_id']}
                            {f"(TL: {ward['timeline_ward_id']})" if ward['timeline_ward_id'] else ""}
                            - <span class="team-{ward['team']}">{ward['team']}</span>
                        </span>
                        <span class="ward-meta">
                            {ward['ward_type']} | Frames: {ward['frame_start']}-{ward['frame_end']} ({duration}f)
                        </span>
                    </div>
                    <div class="ward-content">
'''

            if has_image:
                html += f'''
                        <img class="ward-image" src="data:image/jpeg;base64,{img_data}" alt="Ward #{ward['ward_id']}">
'''

            html += f'''
                        <div class="ward-details">
                            <div class="ward-detail">
                                <div class="ward-detail-label">Position</div>
                                ({ward['x_pixel']}, {ward['y_pixel']})
                            </div>
                            <div class="ward-detail">
                                <div class="ward-detail-label">Class</div>
                                {ward['class_name']}
                            </div>
                            <div class="ward-detail">
                                <div class="ward-detail-label">Ward Type</div>
                                {ward['ward_type']}
                            </div>
                            <div class="ward-detail">
                                <div class="ward-detail-label">Team</div>
                                <span class="team-{ward['team']}">{ward['team']}</span>
                            </div>
                            <div class="ward-detail">
                                <div class="ward-detail-label">Confidence</div>
                                {ward['confidence_avg']:.3f}
                            </div>
                            <div class="ward-detail">
                                <div class="ward-detail-label">Creator ID</div>
                                {ward['creator_id'] or '-'}
                            </div>
                            <div class="ward-detail">
                                <div class="ward-detail-label">Placed (ms)</div>
                                {ward['timestamp_placed'] or '-'}
                            </div>
                        </div>
                    </div>
                </div>
'''

        html += '''
            </div>
        </div>
'''

    html += '''
    </div>

    <script>
        function toggleWard(header) {
            const content = header.nextElementSibling;
            content.classList.toggle('show');
        }

        function toggleSection(btn) {
            const section = btn.closest('.section');
            const contents = section.querySelectorAll('.ward-content');
            const allOpen = Array.from(contents).every(c => c.classList.contains('show'));

            contents.forEach(c => {
                if (allOpen) {
                    c.classList.remove('show');
                } else {
                    c.classList.add('show');
                }
            });

            btn.textContent = allOpen ? 'Expand All' : 'Collapse All';
        }
    </script>
</body>
</html>
'''
    return html


def generate_report(match_id: str, dataset_dir: Path, output_dir: Path):
    """1試合のレポートを生成"""
    match_dir = dataset_dir / match_id
    csv_path = match_dir / "wards_matched.csv"

    if not csv_path.exists():
        print(f"スキップ: {match_id} (wards_matched.csv not found)")
        return None

    print(f"生成中: {match_id}")

    # データ読み込み
    wards = load_wards_matched(csv_path)

    # HTML生成
    html = generate_html_report(match_id, wards, match_dir)

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{match_id}.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  完了: {output_path}")
    return output_path


def generate_index(match_ids: List[str], output_dir: Path):
    """インデックスページを生成"""

    stats = []
    for match_id in match_ids:
        csv_path = DATASET_DIR / match_id / "wards_matched.csv"
        if not csv_path.exists():
            continue
        wards = load_wards_matched(csv_path)
        matched = len([w for w in wards if w['match_status'] == 'matched'])
        timeline_only = len([w for w in wards if w['match_status'] == 'timeline_only'])
        detection_only = len([w for w in wards if w['match_status'] == 'detection_only'])
        total = matched + timeline_only
        rate = matched / total * 100 if total > 0 else 0
        stats.append({
            'match_id': match_id,
            'matched': matched,
            'timeline_only': timeline_only,
            'detection_only': detection_only,
            'total': total,
            'rate': rate
        })

    total_matched = sum(s['matched'] for s in stats)
    total_timeline = sum(s['total'] for s in stats)
    overall_rate = total_matched / total_timeline * 100 if total_timeline > 0 else 0

    html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>Ward Matching Report - Index</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{ color: #00d4ff; }}
        .summary {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
        }}
        .summary-item {{ text-align: center; }}
        .summary-value {{ font-size: 2em; font-weight: bold; color: #00d4ff; }}
        .summary-label {{ color: #888; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #16213e;
            border-radius: 10px;
            overflow: hidden;
        }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #0f3460; color: #00d4ff; }}
        tr:hover {{ background: #1a4a7a; }}
        a {{ color: #00d4ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .rate {{ font-weight: bold; }}
        .rate-high {{ color: #2ecc71; }}
        .rate-mid {{ color: #f39c12; }}
        .rate-low {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <h1>Ward Matching Report Index</h1>

    <div class="summary">
        <div class="summary-item">
            <div class="summary-value">{len(stats)}</div>
            <div class="summary-label">Matches</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{overall_rate:.1f}%</div>
            <div class="summary-label">Overall Rate</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{total_matched}</div>
            <div class="summary-label">Matched</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{total_timeline}</div>
            <div class="summary-label">Timeline Total</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Match ID</th>
                <th>Matched</th>
                <th>Timeline Only</th>
                <th>Detection Only</th>
                <th>Total</th>
                <th>Rate</th>
            </tr>
        </thead>
        <tbody>
'''

    for s in sorted(stats, key=lambda x: x['rate'], reverse=True):
        rate_class = 'rate-high' if s['rate'] >= 90 else 'rate-mid' if s['rate'] >= 80 else 'rate-low'
        html += f'''
            <tr>
                <td><a href="{s['match_id']}.html">{s['match_id']}</a></td>
                <td>{s['matched']}</td>
                <td>{s['timeline_only']}</td>
                <td>{s['detection_only']}</td>
                <td>{s['total']}</td>
                <td class="rate {rate_class}">{s['rate']:.1f}%</td>
            </tr>
'''

    html += '''
        </tbody>
    </table>
</body>
</html>
'''

    output_path = output_dir / "index.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"インデックス生成: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Ward マッチング結果HTMLレポート生成")
    parser.add_argument("--match", type=str, help="試合ID")
    parser.add_argument("--all", action="store_true", help="全試合を処理")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)

    if args.match:
        match_id = args.match if args.match.startswith("JP1-") else f"JP1-{args.match}"
        generate_report(match_id, dataset_dir, output_dir)
    elif args.all:
        match_ids = [d.name for d in sorted(dataset_dir.glob("JP1-*")) if d.is_dir()]
        for match_id in match_ids:
            generate_report(match_id, dataset_dir, output_dir)
        generate_index(match_ids, output_dir)
        print(f"\n全{len(match_ids)}試合のレポートを生成しました")
        print(f"インデックス: {output_dir / 'index.html'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
