import requests
import time
from tqdm import tqdm
from datetime import datetime, timedelta

# ─── API リクエスト関数 ───────────────────────────────────────────
def api_request_with_retry(url, max_retries=3, interval=1.3):
    """
    429エラー時に自動リトライ + 基本待機
    interval: リクエスト間隔（秒）- 2分100件制限を守るため約1.3秒推奨
    """
    for attempt in range(max_retries):
        response = requests.get(url)

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 10))
            print(f"[429] Rate limit hit. Waiting {retry_after}s... (attempt {attempt+1}/{max_retries})")
            time.sleep(retry_after + 1)
            continue

        if response.status_code == 200:
            time.sleep(interval)  # 成功時も待機（Rate Limit対策）
            return response.json()

        # その他のエラー
        print(f"[ERROR] API returned status {response.status_code}")
        time.sleep(interval)
        return None

    print(f"[ERROR] API request failed after {max_retries} retries")
    return None

'''データセット生成器、希望のティアを入力すると、そのティア帯のリプレイを保存してくれる.'''

class DataGenerator(object):

    '''
        api_key : riot api key
        count : match per each player
    '''
    def __init__(self , api_key , count):
        self.api_key = api_key
        self.count = count

    '''
    queue : {RANKED_SOLO_5x5, RANKED_TFT, RANKED_FLEX_SR, RANKED_FLEX_TT}
    tier : {CHALLENGER, GRANDMASTER, MASTER, DIAMOND, PLATINUM, GOLD, SILVER, BRONZE, IRON}   !NOTICE: 'MASTER+' ONLY TAKE DIVISION 'I'
    division : {I, II, III, IV}
    '''
    def get_puuids(self, queue , tier , division): #queue : RANKED_SOLO_5x5 #tier : CHALLENGER(大文字) #division : I ~ IV
        page = 1             #ページ初期値
        summoners_puuids = []       #サモナー名簿
        while True:
            datas = api_request_with_retry(f'https://jp1.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier}/{division}?page={page}&api_key={self.api_key}')

            # APIエラーチェック（リストでない場合はエラーレスポンス）
            if datas is None or not isinstance(datas, list):
                print(f"[ERROR] get_puuids API returned error: {datas}")
                break

            if len(datas) == 0 or page > 300 :
                break
            page = page + 1
            for data in datas:
                summoners_puuids.append(data['puuid'])

        print(f"[INFO] get_puuids: {tier}-{division} から {len(summoners_puuids)} 人のPUUIDを取得しました")
        return summoners_puuids
    
    #SUMMONERID -> PUUID
    def get_puuid(self, summonerId):
        try:
            puuid = requests.get(f'https://jp1.api.riotgames.com/lol/summoner/v4/summoners/{summonerId}?api_key={self.api_key}').json()["puuid"]
            # print(f'puuid : {puuid}')
            return puuid
        except (requests.RequestException, KeyError, ValueError):
            return None
        

    def is_in_recent_patch(self, game_creation_millisec, patch_start_datetime):
        
        dt_obj = datetime.strptime(patch_start_datetime,'%Y.%m.%d')
        patch_start_millisec = dt_obj.timestamp() * 1000

        return patch_start_millisec < game_creation_millisec
    
    
        
    #PUUID -> MATCHID(S)
    def get_matchIds(self, puuid, patch_start_datetime, min_game_duration, max_game_duration):


        def convert_humantime_to_unixtimestamp(humantime): # 2024.09.25 >> 1727265600
            # 文字列をdatetimeオブジェクトに変換
            date_obj = datetime.strptime(humantime, '%Y.%m.%d').replace(hour=12, minute=0, second=0)
            # datetimeオブジェクトをUNIXタイムスタンプ（秒単位）に変換
            unix_timestamp = int(time.mktime(date_obj.timetuple()))
            return unix_timestamp

        def add_days_to_date(date_str, days):
            # 文字列をdatetimeオブジェクトに変換
            date_obj = datetime.strptime(date_str, '%Y.%m.%d')
            # 指定された日数を加算
            new_date_obj = date_obj + timedelta(days=days)
            # 再び文字列形式に変換
            new_date_str = new_date_obj.strftime('%Y.%m.%d')
            return new_date_str

        # 7日追加（LoLのパッチ周期が14日間隔であることを考慮）
        patch_start_datetime_add_7days = add_days_to_date(patch_start_datetime, 7)
        
        # print("?パッチ開始日 : ", patch_start_datetime)
        # print("?パッチ開始日 + 7日 : ", convert_humantime_to_unixtimestamp(patch_start_datetime))
        # print("?パッチ開始日 + 7日 : ", convert_humantime_to_unixtimestamp(patch_start_datetime_add_7days))
        if puuid == None:
            return []
        
        matchIdsOver15 = []
        matchIds_response = api_request_with_retry(f'https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={convert_humantime_to_unixtimestamp(patch_start_datetime)}&queue=420&type=ranked&start=0&count={self.count}&api_key={self.api_key}')
        matchIds_7day_after_response = api_request_with_retry(f'https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={convert_humantime_to_unixtimestamp(patch_start_datetime_add_7days)}&queue=420&type=ranked&start=0&count={self.count}&api_key={self.api_key}')

        # APIエラーチェック（リストでない場合はエラーレスポンス）
        if not isinstance(matchIds_response, list):
            print(f"[ERROR] matchIds API returned error: {matchIds_response}")
            matchIds = []
        else:
            matchIds = matchIds_response

        if not isinstance(matchIds_7day_after_response, list):
            print(f"[ERROR] matchIds_7day_after API returned error: {matchIds_7day_after_response}")
            matchIds_7day_after = []
        else:
            matchIds_7day_after = matchIds_7day_after_response
        
        # print(f'合計 {len(matchIds)}件の試合記録があります。')
        # print(f'合計 {len(matchIds_7day_after)}件の試合記録があります。（パッチ開始日 + 7日）')
        count = 0
        for matchId in matchIds:
            try:
                match_data = api_request_with_retry(f"https://asia.api.riotgames.com/lol/match/v5/matches/{matchId}?api_key={self.api_key}")
                if match_data is None:
                    continue
                gameDuration = round(match_data['info']['gameDuration'])
                gameCreation = round(match_data['info']['gameCreation'])
                if (max_game_duration*60 >= gameDuration >= min_game_duration*60) and (count < self.count) and self.is_in_recent_patch(gameCreation, patch_start_datetime):
                    matchIdsOver15.append(matchId)
                    count = count + 1
                # else: break は削除 - 条件に合わない試合をスキップして続行
            except (requests.RequestException, KeyError, TypeError):
                pass

        for matchId in matchIds_7day_after:
            try:
                match_data = api_request_with_retry(f"https://asia.api.riotgames.com/lol/match/v5/matches/{matchId}?api_key={self.api_key}")
                if match_data is None:
                    continue
                gameDuration = round(match_data['info']['gameDuration'])
                gameCreation = round(match_data['info']['gameCreation'])
                if (max_game_duration*60 >= gameDuration >= min_game_duration*60) and (count < self.count) and self.is_in_recent_patch(gameCreation, patch_start_datetime):
                    matchIdsOver15.append(matchId)
                    count = count + 1
                # else: break は削除 - 条件に合わない試合をスキップして続行
            except (requests.RequestException, KeyError, TypeError):
                pass

        # 重複削除
        matchIdsOver15 = list(set(matchIdsOver15))  

        return matchIdsOver15

    ############## Updated match-history lookup to use PUUID, since summoner-ID queries have been removed from the Riot API.##############
    # #get matchids { precondition : queue(=ランクタイプ) , tier(=ティア), division(=段階, 例:I,II,III,IV) , patch_start_datetime(=パッチ開始日, 例: '2023.10.08')  , min_game_duration : 最小ゲーム進行時間(分)}
    # def get_tier_matchIds(self, queue, tier, division , max_ids, patch_start_datetime, min_game_duration, max_game_duration):
    
    #     # process : queue, tier, division -> summonerId(s)
    #     puuids = self.get_puuids(self, queue , tier , division)[:max_ids] # max_ids分だけ取得
    #     # print("PUUIDS",puuids)
    #     matchIds = []
        
    #     # process : puuid -> matchId(s)
    #     for puuid in tqdm(puuids ,
    #                       desc = 'Gathering puuids by Riot_API from puuids(not summonerIds)... ', ## 進捗率前方出力文章
    #                       ncols = 120, ## 進捗率出力幅調整
    #                       ascii = ' =', 
    #                       leave=True
    #                       ):
    #         try:
    #             matchIds.extend(self.get_matchIds(self, puuid, patch_start_datetime, min_game_duration, max_game_duration))
    #             # print(len(matchIds),"件の試合累積")
    #             time.sleep(0.001)
    #         except:
    #             pass
        
    #     return list(set(matchIds)) # 重複防止
    
    # DataGenerator 内部メソッド - シリアル処理版
    def get_tier_matchIds(
        self,
        queue: str,
        tier: str,
        division: str,
        max_ids: int,
        patch_start_datetime: str,
        min_game_duration: int,
        max_game_duration: int,
        max_workers: int = None  # 後方互換性のため残す（シリアル処理では無視）
    ):
        """
        指定ティア・ディビジョンのmatchIdを取得（シリアル処理）
        Riot API Rate Limit (100リクエスト/2分) を考慮した設計
        注: max_workers は後方互換性のため残していますが、無視されます
        """
        # 1) ティア/ディビジョン → PUUIDリスト
        puuids = self.get_puuids(queue, tier, division)[:max_ids]

        # 2) シリアル処理でmatchId収集
        match_ids = set()
        for puuid in tqdm(puuids,
                          desc=f'[{tier}-{division}] fetch matchIds',
                          ncols=120, ascii=" =",
                          leave=True):
            try:
                ids = self.get_matchIds(
                    puuid,
                    patch_start_datetime,
                    min_game_duration,
                    max_game_duration
                )
                match_ids.update(ids)
            except Exception as e:
                print(f"[ERROR] Exception for puuid {puuid[:20]}...: {e}")

        print(f"[INFO] {tier}-{division}: 合計 {len(match_ids)} 件のユニークなmatchIdを取得しました")
        return list(match_ids)
