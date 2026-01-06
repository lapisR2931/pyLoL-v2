"""
複数試合のUNDEFINED wardTypeを分析するスクリプト
"""

import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_timeline(match_id: str, api_key: str, routing: str = "asia") -> dict:
    """Timeline APIからデータ取得"""
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    headers = {"X-Riot-Token": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None


def get_match_info(match_id: str, api_key: str, routing: str = "asia") -> dict:
    """Match APIからチャンピオン情報を取得"""
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None


def extract_ward_events(timeline_data: dict) -> list[dict]:
    """タイムラインからward設置イベントを抽出"""
    ward_placed = []
    for frame in timeline_data['info']['frames']:
        for event in frame['events']:
            if event['type'] == 'WARD_PLACED':
                creator_id = event.get('creatorId', 0)
                ward_placed.append({
                    'timestamp': event['timestamp'],
                    'creator_id': creator_id,
                    'ward_type': event.get('wardType', 'UNKNOWN'),
                    'team': 'Blue' if creator_id <= 5 else 'Red'
                })
    return ward_placed


def analyze_match(match_id: str, api_key: str) -> dict:
    """1試合を分析"""
    print(f"  Fetching {match_id}...", end=" ")

    # タイムライン取得
    timeline = get_timeline(match_id, api_key)
    if not timeline:
        print("Failed (timeline)")
        return None

    # 試合情報取得
    match_info = get_match_info(match_id, api_key)
    if not match_info:
        print("Failed (match)")
        return None

    # チャンピオン情報をマッピング
    champion_map = {}
    for p in match_info['info']['participants']:
        champion_map[p['participantId']] = {
            'champion': p['championName'],
            'role': p.get('teamPosition', 'Unknown')
        }

    # wardイベント抽出
    ward_events = extract_ward_events(timeline)

    # UNDEFINED wardの分析
    df = pd.DataFrame(ward_events)
    if df.empty:
        print("No wards")
        return None

    undefined_count = len(df[df['ward_type'] == 'UNDEFINED'])
    total_count = len(df)

    # UNDEFINEDを持つプレイヤー
    undefined_players = []
    if undefined_count > 0:
        undefined_df = df[df['ward_type'] == 'UNDEFINED']
        for creator_id, count in undefined_df['creator_id'].value_counts().items():
            champ_info = champion_map.get(creator_id, {'champion': 'Unknown', 'role': 'Unknown'})
            undefined_players.append({
                'creator_id': creator_id,
                'champion': champ_info['champion'],
                'role': champ_info['role'],
                'undefined_count': count
            })

    print(f"OK (UNDEFINED: {undefined_count}/{total_count})")

    return {
        'match_id': match_id,
        'total_wards': total_count,
        'undefined_count': undefined_count,
        'undefined_ratio': undefined_count / total_count if total_count > 0 else 0,
        'undefined_players': undefined_players,
        'champion_map': champion_map
    }


def main():
    # APIキー読み込み
    env_path = Path(__file__).parent.parent / "RiotAPI.env"
    load_dotenv(env_path)
    api_key = os.getenv("RIOT_API_KEY") or os.getenv("API_KEY")

    if not api_key:
        print("Error: API key not found")
        return

    # データセットから試合ID取得
    dataset_dir = Path("C:/dataset_20260105")
    match_ids = [f.name.replace("-", "_") for f in dataset_dir.iterdir() if f.is_dir()]

    print(f"Found {len(match_ids)} matches\n")
    print("=" * 70)
    print("Analyzing matches...")
    print("=" * 70)

    results = []
    for match_id in match_ids[:10]:  # 最初の10試合
        result = analyze_match(match_id, api_key)
        if result:
            results.append(result)

    # 結果サマリー
    print("\n" + "=" * 70)
    print("UNDEFINED Ward Analysis Summary")
    print("=" * 70)

    print(f"\n{'Match ID':<20} {'Total':>6} {'UNDEFINED':>10} {'Ratio':>8}")
    print("-" * 50)

    for r in results:
        print(f"{r['match_id']:<20} {r['total_wards']:>6} {r['undefined_count']:>10} {r['undefined_ratio']:>7.1%}")

    # UNDEFINEDを持つチャンピオン集計
    print("\n" + "=" * 70)
    print("Champions with UNDEFINED wards")
    print("=" * 70)

    champion_undefined = {}
    for r in results:
        for p in r['undefined_players']:
            champ = p['champion']
            if champ not in champion_undefined:
                champion_undefined[champ] = {'count': 0, 'matches': 0, 'roles': set()}
            champion_undefined[champ]['count'] += p['undefined_count']
            champion_undefined[champ]['matches'] += 1
            champion_undefined[champ]['roles'].add(p['role'])

    print(f"\n{'Champion':<15} {'UNDEFINED':>10} {'Matches':>8} {'Roles'}")
    print("-" * 50)

    for champ, data in sorted(champion_undefined.items(), key=lambda x: -x[1]['count']):
        roles = ', '.join(data['roles'])
        print(f"{champ:<15} {data['count']:>10} {data['matches']:>8} {roles}")

    # 結果をCSVとして保存
    output_dir = Path(__file__).parent.parent / "debug"
    output_dir.mkdir(exist_ok=True)

    # 試合ごとの結果
    match_results = []
    for r in results:
        match_results.append({
            'match_id': r['match_id'],
            'total_wards': r['total_wards'],
            'undefined_count': r['undefined_count'],
            'undefined_ratio': r['undefined_ratio']
        })
    pd.DataFrame(match_results).to_csv(output_dir / 'undefined_ward_analysis.csv', index=False)

    # チャンピオン集計
    champ_results = []
    for champ, data in champion_undefined.items():
        champ_results.append({
            'champion': champ,
            'undefined_count': data['count'],
            'matches': data['matches'],
            'roles': ', '.join(data['roles'])
        })
    pd.DataFrame(champ_results).to_csv(output_dir / 'undefined_ward_champions.csv', index=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
