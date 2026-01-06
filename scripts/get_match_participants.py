"""
試合のparticipant情報（チャンピオン名）を取得するスクリプト
"""

import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv

def get_match_info(match_id: str, api_key: str, routing: str = "asia") -> dict:
    """Match APIからチャンピオン情報を含む試合データを取得"""
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": api_key}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def print_participants(match_data: dict):
    """参加者情報を表示"""
    if not match_data:
        return

    info = match_data.get('info', {})
    participants = info.get('participants', [])

    print("=" * 70)
    print(f"Match ID: {match_data.get('metadata', {}).get('matchId', 'Unknown')}")
    print(f"Game Duration: {info.get('gameDuration', 0) // 60}m {info.get('gameDuration', 0) % 60}s")
    print("=" * 70)

    print("\n【Participants】")
    print(f"{'ID':>3} | {'Team':>4} | {'Role':<10} | {'Champion':<15} | {'Summoner Name'}")
    print("-" * 70)

    for p in participants:
        participant_id = p.get('participantId', 0)
        team = 'Blue' if p.get('teamId') == 100 else 'Red'
        role = p.get('teamPosition', 'Unknown')
        champion = p.get('championName', 'Unknown')
        summoner = p.get('summonerName', 'Unknown')

        print(f"{participant_id:>3} | {team:>4} | {role:<10} | {champion:<15} | {summoner}")

    # 勝敗
    print("\n【Result】")
    for team in info.get('teams', []):
        team_name = 'Blue' if team.get('teamId') == 100 else 'Red'
        result = 'Win' if team.get('win') else 'Lose'
        print(f"  {team_name}: {result}")


def main():
    # .envファイルからAPIキーを読み込み
    env_path = Path(__file__).parent.parent / "RiotAPI.env"
    load_dotenv(env_path)

    api_key = os.getenv("RIOT_API_KEY") or os.getenv("API_KEY")

    if not api_key:
        print("Error: API key not found in RiotAPI.env")
        print("Please set RIOT_API_KEY or API_KEY in RiotAPI.env")
        return

    # 対象のmatchId
    match_id = "JP1_555621265"

    print(f"Fetching match data for: {match_id}")
    match_data = get_match_info(match_id, api_key)

    if match_data:
        # 結果を表示
        print_participants(match_data)

        # JSONとして保存
        output_dir = Path(__file__).parent.parent / "data" / "match"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{match_id}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(match_data, f, indent=2, ensure_ascii=False)

        print(f"\nMatch data saved to: {output_path}")


if __name__ == '__main__':
    main()
