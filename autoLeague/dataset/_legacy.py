"""RiotAPI レガシーメソッド群

旧RiotAPIクラスからMatchFetcherに移行されなかったメソッドをそのまま保持する。
新規コードではこのモジュールを直接使用しないことを推奨。
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from autoLeague.base import BaseRiotClient, RiotApiConfig


class RiotAPILegacy(BaseRiotClient):
    """旧RiotAPIクラスのレガシーメソッドを保持するクラス"""

    def __init__(self, api_key: str, region: str = "JP1") -> None:
        config = RiotApiConfig(api_key=api_key, region=region)
        super().__init__(config)

    # ------------------------------------------------------------------
    # backward compat properties
    # ------------------------------------------------------------------
    @property
    def api_key(self) -> str:
        return self._config.api_key

    @property
    def region(self) -> str:
        return self._config.region

    @property
    def routing(self) -> str:
        return self._config.routing

    @property
    def region_prefix(self) -> str:
        return self._config.region_prefix

    # Accumulates each element of the list sequentially
    @staticmethod
    def transformList(original_list):
        original_list = original_list
        new_list = []

        running_sum = 0
        for value in original_list:
            running_sum += value
            new_list.append(running_sum)

        return new_list

    # Returns the difference of two lists (Blue team - Red team)
    @staticmethod
    def subtractList(blue_list1, red_list2):

        result = [a - b for a, b in zip(blue_list1, red_list2)]

        return result

    def getMatchData(self, matchId):
        data = requests.get(f"https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/{matchId}/timeline?api_key={self._config.api_key}").json()
        #### Blue team participantId 1~5 // Red team participantId 6~10
        BLUE_IDs = [1,2,3,4,5]
        RED_IDs = [6,7,8,9,10]
        #### When frame index is N, the actual time range it contains is from N-1 min 0sec to N-1 min 59sec
        #### Therefore, info from 5min to 15min 0sec is in frames 6 to 15.
        frame_5_to_15 = data['info']['frames'][6:16]

        # (In this time period) Per-team kills, assists, accumulated large object takedowns
        BLUE_KILLS,BLUE_DEATHS,BLUE_ASSISTS,BLUE_OBJECTS,BLUE_GOLDS,BLUE_EXPERIENCES  = [],[],[],[],[],[]   # Note: here we sum assists of all champions
        RED_KILLS,RED_DEATHS,RED_ASSISTS,RED_OBJECTS,RED_GOLDS,RED_EXPERIENCES = [],[],[],[],[],[]



        for frame in frame_5_to_15:

            blue_kills,blue_deaths,blue_assists,blue_objects,blue_golds,blue_experiences  = 0,0,0,0,0,0
            red_kills,red_deaths,red_assists,red_objects,red_golds,red_experiences = 0,0,0,0,0,0


            for event in frame['events']:

                if event['type'] == "CHAMPION_KILL":            # KILL, DEATH, ASSIST
                    if event['killerId'] in BLUE_IDs:
                        blue_kills += 1
                    if event['killerId'] in RED_IDs:
                        red_kills += 1
                    if event['victimId'] in BLUE_IDs:
                        blue_deaths += 1
                    if event['victimId'] in RED_IDs:
                        red_deaths += 1
                    if event.get("assistingParticipantIds") != None:
                        for assistId in event.get("assistingParticipantIds"):
                            if assistId in BLUE_IDs:
                                blue_assists += 1
                            if assistId in RED_IDs:
                                red_assists += 1

                if event['type'] == "ELITE_MONSTER_KILL":       # ELITE MONSTER KILL
                    if event['killerTeamId'] == 100:
                        blue_objects += 1
                    else:
                        red_objects += 1


            for index in frame['participantFrames']:
                if int(index)  <= 5:
                    #print(frame['participantFrames'][index]['totalGold'])
                    blue_golds += frame['participantFrames'][index]['totalGold']
                    blue_experiences += frame['participantFrames'][str(index)]['xp']
                else:
                    red_golds += frame['participantFrames'][index]['totalGold']
                    red_experiences += frame['participantFrames'][index]['xp']




            BLUE_KILLS.append(blue_kills)
            BLUE_DEATHS.append(blue_deaths)
            BLUE_ASSISTS.append(blue_assists)
            BLUE_OBJECTS.append(blue_objects)
            BLUE_GOLDS.append(blue_golds)
            BLUE_EXPERIENCES.append(blue_experiences)

            RED_KILLS.append(red_kills)
            RED_DEATHS.append(red_deaths)
            RED_ASSISTS.append(red_assists)
            RED_OBJECTS.append(red_objects)
            RED_GOLDS.append(red_golds)
            RED_EXPERIENCES.append(red_experiences)

        # Convert to cumulative count (AFTER the loop, only once)
        BLUE_KILLS = self.transformList(BLUE_KILLS)
        BLUE_DEATHS = self.transformList(BLUE_DEATHS)
        BLUE_ASSISTS = self.transformList(BLUE_ASSISTS)
        BLUE_OBJECTS = self.transformList(BLUE_OBJECTS)
        BLUE_GOLDS = self.transformList(BLUE_GOLDS)
        BLUE_EXPERIENCES = self.transformList(BLUE_EXPERIENCES)

        RED_KILLS = self.transformList(RED_KILLS)
        RED_DEATHS = self.transformList(RED_DEATHS)
        RED_ASSISTS = self.transformList(RED_ASSISTS)
        RED_OBJECTS = self.transformList(RED_OBJECTS)
        RED_GOLDS = self.transformList(RED_GOLDS)
        RED_EXPERIENCES = self.transformList(RED_EXPERIENCES)

        # Calculate team difference
        DIFF_KILLS = self.subtractList(BLUE_KILLS, RED_KILLS)
        DIFF_DEATHS = self.subtractList(BLUE_DEATHS, RED_DEATHS)
        DIFF_ASSISTS = self.subtractList(BLUE_ASSISTS, RED_ASSISTS)
        DIFF_OBJECTS = self.subtractList(BLUE_OBJECTS, RED_OBJECTS)
        DIFF_GOLDS = self.subtractList(BLUE_GOLDS, RED_GOLDS)
        DIFF_EXPERIENCES = self.subtractList(BLUE_EXPERIENCES, RED_EXPERIENCES)

        allData = []
        '''allData.append(matchId)
        allData.extend(BLUE_KILLS) , allData.extend(BLUE_DEATHS) , allData.extend(BLUE_ASSISTS) , allData.extend(BLUE_OBJECTS) , allData.extend(BLUE_GOLDS) , allData.extend(BLUE_EXPERIENCES)
        allData.extend(RED_KILLS) , allData.extend(RED_DEATHS) , allData.extend(RED_ASSISTS) , allData.extend(RED_OBJECTS) , allData.extend(RED_GOLDS) , allData.extend(RED_EXPERIENCES)'''

        allData.append(DIFF_KILLS) , allData.append(DIFF_DEATHS) , allData.append(DIFF_ASSISTS) , allData.append(DIFF_OBJECTS) , allData.append(DIFF_GOLDS) , allData.append(DIFF_EXPERIENCES)

        # Convert the 2D list to a NumPy array and transpose it to have shape (time_steps, features)
        data_np = np.array(allData).transpose()

        # Add an extra dimension for samples, resulting in shape (samples, time_steps, features)
        data_np = np.expand_dims(data_np, axis=0)
        return data_np



    def writeCSVfile(self, matchIds): # Save the extracted info to CSV file. Process for all matchIds.

        features = ['matchID']
        for i in range(10):
            features.append(f'blue_kills{i}')

        for i in range(10):
            features.append(f'blue_deaths{i}')

        for i in range(10):
            features.append(f'blue_assists{i}')

        for i in range(10):
            features.append(f'blue_objects{i}')

        for i in range(10):
            features.append(f'red_kills{i}')

        for i in range(10):
            features.append(f'red_deaths{i}')

        for i in range(10):
            features.append(f'red_assists{i}')

        for i in range(10):
            features.append(f'red_objects{i}')


        f = open(rf'riot_api_dataset.csv','a', newline='')
        wr = csv.writer(f)
        wr.writerow(features)

        for matchId in tqdm(matchIds):
            row = self.getMatchData(matchId)

            wr.writerow(row)

        f.close()

        print('csv write complete')


    @staticmethod
    def filter_matches_by_excluded_champions(csv_in, csv_out, excluded_champions):
        # Load CSV
        df = pd.read_csv(csv_in)

        # Select only champion_* columns and check if any row contains
        #    EXCLUDED_CHAMPIONS (True if at least one exists)
        champ_cols = [c for c in df.columns if c.startswith("champion_")]
        has_excluded = df[champ_cols].isin(excluded_champions).any(axis=1)

        # Invert condition (~) -> Select only matches WITHOUT excluded champions
        filtered_df = df.loc[~has_excluded, ["match_id"]]

        # Save
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        filtered_df.to_csv(csv_out, index=False)

        print(f"Filtered {len(filtered_df)} / {len(df)} match_ids saved to {csv_out}")

    def save_top_matches_by_score(
        self,
        avg_score_csv_path: str,
        data_json_path: str,
        output_path: str,
        top_n: int = 6000,
    ):
        """
        Extract top_n match_ids from avg_score_csv_path based on avg_score,
        filter to only matches existing in data_json_path (kill_events_timeline JSON),
        and save to output_path.
        """
        import csv, json, os

        # Load match_id, avg_score from average score CSV
        matches = []
        with open(avg_score_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m_id = row["match_id"]
                if m_id.startswith(self._config.region_prefix):
                    m_id = m_id.replace(self._config.region_prefix, "")
                avg_score = float(row["avg_score"])
                matches.append((m_id, avg_score))

        # Load data.json
        with open(data_json_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        orig_inner_data = original_data.get("data", {})
        existing_ids = set(orig_inner_data.keys())

        # Extract top N matches by avg_score from match_ids existing in JSON
        matches_in_json = [(m_id, score) for (m_id, score) in matches if m_id in existing_ids]
        matches_in_json.sort(key=lambda x: x[1], reverse=True)
        top_match_ids = [m_id for (m_id, _) in matches_in_json[:top_n]]

        # Filter and save
        filtered_dict = {m_id: orig_inner_data[m_id] for m_id in top_match_ids}
        result_obj = {"data": filtered_dict}

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_obj, f, ensure_ascii=False, indent=2)

        print(f"Filtered top {len(top_match_ids)} matches based on avg_score.")
        print(f"Output: {output_path}")


    @staticmethod
    def find_disjoint_matches_combinations(target_matches, max_solutions):

        def load_matches(csv_path):
            matches = []
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                for row in reader:
                    match_id = row[0]
                    champions = row[1:]
                    matches.append((match_id, set(champions)))
            return matches

        def find_disjoint_matches_all(matches, target, start, used_champions, selected, solutions, max_solutions=3):
            # Save one combination when target number of matches is selected
            if len(selected) == target:
                solutions.append(selected.copy())
                return
            # Iterate through remaining matches
            for i in range(start, len(matches)):
                # Exit if desired number of combinations found
                if len(solutions) >= max_solutions:
                    break
                match_id, champs = matches[i]
                # Check if overlaps with champions from selected matches
                if used_champions & champs:
                    continue
                selected.append(match_id)
                new_used = used_champions.union(champs)
                find_disjoint_matches_all(matches, target, i+1, new_used, selected, solutions, max_solutions)
                selected.pop()

        # CSV file path
        csv_file = os.path.join('data', 'match_champions.csv')
        matches = load_matches(csv_file)
        solutions = []
        find_disjoint_matches_all(matches, target_matches, 0, set(), [], solutions, max_solutions)

        if solutions:
            print(f"Found {len(solutions)} sets of {target_matches} matches with no overlapping champions:")
            for sol in solutions:
                print(sol)
        else:
            print("No matching sets found.")




    @staticmethod
    def find_disjoint_matches_combinations_with_initial(
        target_matches,             # Desired final match_id count
        max_solutions,              # Maximum number of different combinations to find
        initial_matches,            # Already non-overlapping match_id list (e.g., ['KR-123', ...])
        csv_file_path=None          # Match CSV file path (default None -> uses data folder)
    ):
        if csv_file_path is None:
            csv_file_path = os.path.join('data', 'match_champions.csv')


        def load_matches(csv_path):
            """Load from CSV as (match_id, set of champions appearing in that match)"""
            matches = []
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                for row in reader:
                    match_id = row[0]
                    champions = row[1:]
                    matches.append((match_id, set(champions)))
            return matches

        def find_disjoint_matches_additional(matches, target, start_index,
                                            used_champions, selected,
                                            solutions, max_solutions):
            """
            Assumes some match_ids are already in selected,
            and used_champions contains all champions from selected matches.
            """
            # If current selected count reaches target, one combination is complete, so save it
            if len(selected) == target:
                solutions.append(selected.copy())
                return

            for i in range(start_index, len(matches)):
                # Stop if we've found enough combinations
                if len(solutions) >= max_solutions:
                    break

                match_id, champs = matches[i]

                # Skip match_ids already in the initial list to avoid duplicates
                if match_id in selected:
                    continue

                # Check if there are overlapping champions
                if used_champions & champs:
                    continue

                # Select new match
                selected.append(match_id)
                new_used = used_champions.union(champs)

                # Search next matches (from i+1)
                find_disjoint_matches_additional(matches, target, i+1,
                                                new_used, selected,
                                                solutions, max_solutions)

                # Backtracking (restore)
                selected.pop()

        # Load CSV
        matches = load_matches(csv_file_path)

        # Get champion set from already given match_ids
        used_champions = set()
        for match_id, champ_set in matches:
            if match_id in initial_matches:
                used_champions |= champ_set

        # Number of match_ids selected so far
        current_len = len(initial_matches)

        # If already has target_matches or more, return as one solution
        if current_len >= target_matches:
            return [initial_matches]

        # Number of additional matches needed via recursion
        missing = target_matches - current_len

        solutions = []

        # Start with initial_matches in selected
        selected = initial_matches.copy()

        # Recursive call
        find_disjoint_matches_additional(matches,
                                        target_matches,
                                        start_index=0,
                                        used_champions=used_champions,
                                        selected=selected,
                                        solutions=solutions,
                                        max_solutions=max_solutions)

        return solutions

    def find_matches_combinations_with_initial_allow_duplicate_2(
            self,
            distinct_champion_target=170,   # Target number of distinct champions
            max_solutions=10,               # Maximum number of solutions to return
            initial_matches=None,           # List of already selected matches
            csv_file_path=None              # CSV path
    ):
        if initial_matches is None:
            initial_matches = []
        if csv_file_path is None:
            csv_file_path = os.path.join('data', 'match_champions.csv')

        # Load CSV
        def load_matches(csv_path):
            matches = []
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    match_id = row[0]
                    champions = set(row[1:])
                    matches.append((match_id, champions))
            return matches

        # API call (only when not in CSV)
        def get_champnames_per_match(matchid):
            url = f'https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/{matchid}?api_key={self._config.api_key}'
            data = requests.get(url, timeout=3).json()
            return [p['championName'] for p in data['info']['participants']]

        matches = load_matches(csv_file_path)
        csv_ids = {mid for mid, _ in matches}

        # Prepare initial used_champions
        used_champs = set()
        for mid in initial_matches:
            if mid in csv_ids:
                used_champs |= dict(matches)[mid]
            else:
                if not self._config.api_key:
                    raise ValueError(f"API key required for match {mid}")
                used_champs |= set(get_champnames_per_match(mid))

        # Target achieved with initial set alone?
        if len(used_champs) >= distinct_champion_target:
            print(f"Initial combination already satisfies: {len(initial_matches)} matches, {len(used_champs)} unique champions")
            return [initial_matches]

        solutions = []
        best_len = float('inf')

        # Stack: (start_idx, current_union, selected_matches)
        stack = [(0, used_champs, initial_matches.copy())]

        while stack and len(solutions) < max_solutions:
            start, union_champs, selected = stack.pop()

            # Target achieved
            if len(union_champs) >= distinct_champion_target:
                # Record solution
                if len(selected) < best_len:
                    best_len = len(selected)
                    solutions.clear()
                if len(selected) == best_len:
                    solutions.append(selected.copy())
                continue

            # Pruning: no need to check if already larger than current best_len
            if len(selected) >= best_len:
                continue

            # Next candidates
            for i in range(start, len(matches)):
                mid, champs = matches[i]
                if mid in selected:
                    continue
                # Number of overlapping champions
                overlap = len(union_champs & champs)
                if overlap > 5:
                    continue  # Skip matches with 6+ overlaps

                new_union = union_champs | champs
                new_selected = selected + [mid]
                stack.append((i+1, new_union, new_selected))

        # Output results
        if not solutions:
            print("No matching combination found.")
            return []

        final = []
        for idx, sol in enumerate(solutions, 1):
            # Calculate unique champion count
            cu = set()
            for m in sol:
                if m in csv_ids:
                    cu |= dict(matches)[m]
                else:
                    cu |= set(get_champnames_per_match(m))
            print(f"[Solution {idx}] matches={len(sol)}, unique champions={len(cu)} -> {sol}")
            final.append(sol)

        return final



    def find_matches_combinations_with_initial_allow_duplicate(self,
            distinct_champion_target=170,   # Target number of distinct champions
            max_solutions=10,               # Maximum number of solutions to return
            initial_matches=None,           # List of already selected matches
            csv_file_path=None              # CSV file path
    ):
        import os, csv, requests

        if initial_matches is None:
            initial_matches = []
        if csv_file_path is None:
            csv_file_path = os.path.join('data', 'match_champions.csv')

        # Load CSV
        def load_matches(csv_path):
            matches = []
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # header skip
                for row in reader:
                    match_id = row[0]
                    champions = set(row[1:])
                    matches.append((match_id, champions))
            return matches

        # API call (when needed)
        def get_champnames_per_match(matchid):
            url = f'https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/{matchid}?api_key={self._config.api_key}'
            data = requests.get(url, timeout=3).json()
            return [p['championName'] for p in data['info']['participants']]

        # Load match data
        matches = load_matches(csv_file_path)
        id2champs = {mid: champs for mid, champs in matches}
        csv_ids = set(id2champs.keys())

        # Build used_champions from initial_matches
        used_champions = set()
        for mid in initial_matches:
            if mid in csv_ids:
                used_champions |= id2champs[mid]
            else:
                if not self._config.api_key:
                    raise ValueError(f"API Key required: to query match '{mid}' not in CSV")
                used_champions |= set(get_champnames_per_match(mid))

        # Target achieved with initial set
        if len(used_champions) >= distinct_champion_target:
            print(f"Initial matches sufficient: {len(initial_matches)} matches, {len(used_champions)} unique champions")
            return [initial_matches]

        solutions = []
        best_len = float('inf')
        stack = [(0, used_champions, initial_matches.copy())]

        # DFS + Backtracking
        while stack and len(solutions) < max_solutions:
            start_index, curr_union, selected = stack.pop()

            # Target achieved
            if len(curr_union) >= distinct_champion_target:
                if len(selected) < best_len:
                    best_len = len(selected)
                    solutions.clear()
                if len(selected) == best_len:
                    solutions.append(selected.copy())
                continue

            # Pruning
            if len(selected) >= best_len:
                continue

            # Next candidates
            for i in range(start_index, len(matches)):
                mid, champs = matches[i]
                if mid in selected:
                    continue
                new_union = curr_union | champs
                new_selected = selected + [mid]
                stack.append((i+1, new_union, new_selected))

        # Print results (match count, unique champion count) and return
        if not solutions:
            print("No matching combination found.")
            return []

        for idx, sol in enumerate(solutions, 1):
            champ_set = set()
            for m in sol:
                if m in csv_ids:
                    champ_set |= id2champs[m]
                else:
                    champ_set |= set(get_champnames_per_match(m))
            print(f"[Solution {idx}] matches={len(sol)}, unique champions={len(champ_set)} -> {sol}")

        return solutions


    def find_disjoint_matches_combinations_with_initial_ver2(
            self,
            target_matches: int,
            max_solutions: int,
            initial_matches: list[str],
            csv_file_path: str | None = None,
            max_duplicates: int = 0              # allow up to n duplicates
    ):

        if csv_file_path is None:
            csv_file_path = os.path.join('data', 'match_champions.csv')
        if initial_matches is None:
            initial_matches = []

        # Load CSV
        with open(csv_file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            matches = [(row[0], set(row[1:])) for row in reader]

        id2champs = {mid: champs for mid, champs in matches}
        csv_ids   = set(id2champs.keys())

        # API Helper
        _api_cache: dict[str, set[str]] = {}

        def champs_from_api(matchid: str) -> set[str]:
            if matchid in _api_cache:
                return _api_cache[matchid]
            if not self._config.api_key:
                raise ValueError("API key is required")
            url = f"https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/{matchid}?api_key={self._config.api_key}"
            data = requests.get(url, timeout=3).json()
            names = {p["championName"] for p in data["info"]["participants"]}
            _api_cache[matchid] = names
            return names

        # Initial State
        used = set()
        for mid in initial_matches:
            used |= id2champs[mid] if mid in csv_ids else champs_from_api(mid)

        # If initial combination has target number of matches or more
        if len(initial_matches) >= target_matches:
            print(f"[Initial] matches={len(initial_matches)}, unique champions={len(used)}")
            return [initial_matches]

        # Sort by contribution of new champions
        sorted_matches = sorted(matches,
                                key=lambda mc: len(mc[1] - used),
                                reverse=True)

        solutions: list[list[str]] = []

        # DFS (allow n duplicates)
        def dfs(start: int, union: set, chosen: list, dupes_used: int):
            if len(chosen) == target_matches:
                solutions.append(chosen.copy())
                return
            if len(solutions) >= max_solutions:
                return

            for idx in range(start, len(sorted_matches)):
                mid, champs = sorted_matches[idx]
                if mid in chosen:
                    continue

                overlap = union & champs
                new_dupes = len(overlap)
                if dupes_used + new_dupes > max_duplicates:
                    continue  # Exceeds allowed duplicates

                chosen.append(mid)
                dfs(idx + 1, union | champs, chosen, dupes_used + new_dupes)
                chosen.pop()

                if len(solutions) >= max_solutions:
                    return

        dfs(0, used, initial_matches.copy(), dupes_used=0)

        # Results
        if not solutions:
            print("No matching combination found.")
            return []

        for i, combo in enumerate(solutions, 1):
            champ_set = set()
            for m in combo:
                champ_set |= id2champs[m] if m in csv_ids else champs_from_api(m)
            print(f"[Solution {i}] matches={len(combo)}, unique champions={len(champ_set)} "
                f"(duplicates allowed: {max_duplicates}) -> {combo}")

        return solutions


    def find_disjoint_matches_combinations_with_initial_ver3(
            self,
            target_matches: int,
            max_solutions: int,
            initial_matches: list[str] | None = None,
            csv_file_path: str | None = None,
            max_duplicates: int = 0,
            verbose: bool = True,
    ):
        """
        Find match combinations of target_matches allowing up to max_duplicates duplicate champions.
        """
        import os, csv, requests, itertools

        if csv_file_path is None:
            csv_file_path = os.path.join("data", "match_champions.csv")
        if initial_matches is None:
            initial_matches = []

        # Load CSV + Count all unique champions
        with open(csv_file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # header skip
            matches = [(row[0], set(row[1:])) for row in reader]

        all_unique_champs = set().union(*(ch for _, ch in matches))
        if verbose:
            print(f"Total unique champions across all CSV matches: {len(all_unique_champs)}\n")

        id2champs = {mid: champs for mid, champs in matches}
        csv_ids   = set(id2champs.keys())

        # API helper (handle matches not in CSV)
        _api_cache: dict[str, set[str]] = {}

        def champs_from_api(matchid: str) -> set[str]:
            if matchid in _api_cache:
                return _api_cache[matchid]
            if not self._config.api_key:
                raise ValueError("self.api_key is not set.")
            url = f"https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/{matchid}?api_key={self._config.api_key}"
            data = requests.get(url, timeout=3).json()
            names = {p["championName"] for p in data["info"]["participants"]}
            _api_cache[matchid] = names
            return names

        # Initial State
        used = set()
        for mid in initial_matches:
            used |= id2champs[mid] if mid in csv_ids else champs_from_api(mid)

        if len(initial_matches) >= target_matches:
            print(f"[Initial] matches={len(initial_matches)}, unique champions={len(used)}")
            return [initial_matches]

        # Sort by contribution of new champions (descending)
        sorted_matches = sorted(
            matches,
            key=lambda mc: len(mc[1] - used),
            reverse=True,
        )

        # DFS Search
        solutions: list[list[str]] = []
        node_counter = itertools.count(1)  # Visited node counter

        def dfs(start: int, union: set, chosen: list, dupes_used: int, depth: int):
            # Print progress every 1000 nodes
            if verbose and next(node_counter) % 1000 == 0:
                print(f"depth={depth:<2} | nodes searched={next(node_counter)-1:<7} "
                    f"| selected={len(chosen)} | duplicates used={dupes_used}")

            if len(chosen) == target_matches:
                solutions.append(chosen.copy())
                if verbose:
                    print(f"Solution found! ({len(chosen)} matches)")
                return
            if len(solutions) >= max_solutions:
                return

            for idx in range(start, len(sorted_matches)):
                mid, champs = sorted_matches[idx]
                if mid in chosen:
                    continue

                overlap_cnt = len(union & champs)
                if dupes_used + overlap_cnt > max_duplicates:
                    continue  # Exceeds allowed duplicates

                chosen.append(mid)
                dfs(idx + 1,
                    union | champs,
                    chosen,
                    dupes_used + overlap_cnt,
                    depth + 1)
                chosen.pop()

                if len(solutions) >= max_solutions:
                    return

        dfs(0, used, initial_matches.copy(), dupes_used=0, depth=0)

        # Output Results
        if not solutions:
            print("No matching combination found.")
            return []

        for i, combo in enumerate(solutions, 1):
            champ_set = set()
            for m in combo:
                champ_set |= id2champs[m] if m in csv_ids else champs_from_api(m)
            print(f"\n[Solution {i}] "
                f"matches={len(combo)}, unique champions={len(champ_set)} "
                f"(duplicates allowed: {max_duplicates})\n-> {combo}")

        return solutions



    def find_disjoint_matches_combinations_with_initial_ver4(
            self,
            target_matches: int,
            max_solutions: int,
            initial_matches: list[str] | None = None,
            csv_file_path: str | None = None,
            max_duplicates: int = 0,
            verbose: bool = True,
    ):
        """
        Allow up to max_duplicates duplicate champions.
        Returns: [(match_id_list, unique_champion_set, coverage_ratio), ...]
        """
        import os, csv, requests, itertools

        if csv_file_path is None:
            csv_file_path = os.path.join("data", "match_champions.csv")
        if initial_matches is None:
            initial_matches = []

        # Load CSV + Count all unique champions
        with open(csv_file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            matches = [(row[0], set(row[1:])) for row in reader]

        all_unique_champs = set().union(*(ch for _, ch in matches))
        if verbose:
            print(f"Total unique champions across all CSV matches: {len(all_unique_champs)}\n")

        id2champs = {mid: champs for mid, champs in matches}
        csv_ids   = set(id2champs.keys())

        # API Helper
        _api_cache: dict[str, set[str]] = {}

        def champs_from_api(matchid: str) -> set[str]:
            if matchid in _api_cache:
                return _api_cache[matchid]
            if not self._config.api_key:
                raise ValueError("self.api_key is required.")
            url = (f"https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/"
                f"{matchid}?api_key={self._config.api_key}")
            data = requests.get(url, timeout=3).json()
            names = {p["championName"] for p in data["info"]["participants"]}
            _api_cache[matchid] = names
            return names

        # Initial State
        used = set()
        for mid in initial_matches:
            used |= id2champs[mid] if mid in csv_ids else champs_from_api(mid)

        if len(initial_matches) >= target_matches:
            print(f"[Initial] matches={len(initial_matches)}, unique champions={len(used)}")
            champ_set = used
            coverage = sum(1 for _, ch in matches if ch <= champ_set) / len(matches) * 100
            return [(initial_matches, champ_set, coverage)]

        sorted_matches = sorted(
            matches,
            key=lambda mc: len(mc[1] - used),
            reverse=True,
        )

        # DFS Search
        solutions: list[list[str]] = []
        node_counter = itertools.count(1)

        def dfs(start: int, union: set, chosen: list, dupes_used: int, depth: int):
            if verbose and next(node_counter) % 1000 == 0:
                print(f"depth={depth:<2} | nodes={next(node_counter)-1:<7} "
                    f"| selected={len(chosen)} | duplicates={dupes_used}")
            if len(chosen) == target_matches:
                solutions.append(chosen.copy())
                if verbose:
                    print(f"Solution found! ({len(chosen)} matches)")
                return
            if len(solutions) >= max_solutions:
                return

            for idx in range(start, len(sorted_matches)):
                mid, champs = sorted_matches[idx]
                if mid in chosen:
                    continue
                overlap = len(union & champs)
                if dupes_used + overlap > max_duplicates:
                    continue
                chosen.append(mid)
                dfs(idx+1, union | champs, chosen,
                    dupes_used + overlap, depth+1)
                chosen.pop()
                if len(solutions) >= max_solutions:
                    return

        dfs(0, used, initial_matches.copy(), dupes_used=0, depth=0)

        # Results
        if not solutions:
            print("No matching combination found.")
            return []

        final_results = []

        for i, combo in enumerate(solutions, 1):
            champ_set = set()
            for m in combo:
                champ_set |= id2champs[m] if m in csv_ids else champs_from_api(m)
            # Coverage: ratio of CSV matches fully covered by champ_set
            covered = sum(1 for _, ch in matches if ch <= champ_set)
            coverage_ratio = covered / len(matches) * 100

            print(f"\n[Solution {i}] "
                  f"matches={len(combo)}, unique champions={len(champ_set)} "
                  f"(duplicates allowed: {max_duplicates})")
            print(f"  > {covered}/{len(matches)} CSV matches "
                  f"({coverage_ratio:.2f}%) are covered by this champion set.")
            print(f"  > match_id list: {combo}")

            # Calculate and print 'newly added champions for each match'
            print("  > New champions per match:")
            seen = set()
            for mid in combo:
                champs = id2champs[mid] if mid in csv_ids else champs_from_api(mid)
                new_champs = champs - seen          # Champions newly added by this match
                seen |= champs
                new_list = ", ".join(sorted(new_champs)) if new_champs else "(none)"
                print(f"     - {mid}: {new_list}")

            final_results.append((combo, champ_set, coverage_ratio))

        return final_results



    def calc_coverage_with_initial_matches(
            self,
            initial_matches: list[str],
            csv_file_path: str | None = None,
            verbose: bool = True
    ):
        """
        Calculate what percentage of all matches in csv_file_path
        are covered by the unique champion set from initial_matches.

        Returns: (unique_champ_set, coverage_ratio, covered_count, total_matches)
        """
        import os, csv, requests

        if csv_file_path is None:
            csv_file_path = os.path.join("data", "match_champions.csv")
        if initial_matches is None:
            initial_matches = []

        # Load CSV
        with open(csv_file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # header skip
            matches = [(row[0], set(row[1:])) for row in reader]

        id2champs = {mid: champs for mid, champs in matches}
        csv_ids   = set(id2champs.keys())

        # API Helper (matches not in CSV)
        _api_cache: dict[str, set[str]] = {}

        def champs_from_api(matchid: str) -> set[str]:
            if matchid in _api_cache:
                return _api_cache[matchid]
            if not self._config.api_key:
                raise ValueError("self.api_key is required.")
            url = (f"https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/"
                f"{matchid}?api_key={self._config.api_key}")
            data = requests.get(url, timeout=3).json()
            names = {p["championName"] for p in data["info"]["participants"]}
            _api_cache[matchid] = names
            return names

        # initial_matches -> champ_set
        champ_set = set()
        for mid in initial_matches:
            champ_set |= id2champs[mid] if mid in csv_ids else champs_from_api(mid)

        # Calculate coverage
        total = len(matches)
        covered = sum(1 for _, champs in matches if champs <= champ_set)
        coverage_ratio = covered / total * 100 if total else 0.0

        if verbose:
            print(f"> Initial match count     : {len(initial_matches)}")
            print(f"> Unique champion count   : {len(champ_set)}")
            print(f"> Total CSV matches       : {total}")
            print(f"> Fully covered matches   : {covered}")
            print(f"> Coverage                : {coverage_ratio:.2f}%")

        return champ_set, coverage_ratio, covered, total



    def find_best_coverage_with_additional_matches(
            self,
            initial_matches: list[str],      # Fixed match_id list
            add_matches: list[str],          # Additional candidate match_id list
            n: int,                          # Number to select from add_matches
            csv_file_path: str | None = None,
            top_k: int | None = 100,         # Brute force only top k candidates when many (None for all)
    ):
        """
        Keep initial_matches fixed, add n from add_matches
        to find combination that maximizes CSV match coverage.
        Returns: (best_combo, coverage_ratio, covered_cnt, total_cnt)
        """
        import os, csv, itertools, requests

        if csv_file_path is None:
            csv_file_path = os.path.join("data", "match_champions.csv")

        # Load CSV
        with open(csv_file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            csv_matches = [(row[0], set(row[1:])) for row in reader]

        id2champs = {mid: champs for mid, champs in csv_matches}
        csv_ids   = set(id2champs.keys())

        # API Helper
        _cache: dict[str, set[str]] = {}

        def champs_api(mid):
            if mid in _cache:
                return _cache[mid]
            url = f"https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/{mid}?api_key={self._config.api_key}"
            data = requests.get(url, timeout=3).json()
            names = {p["championName"] for p in data["info"]["participants"]}
            _cache[mid] = names
            return names

        # Initial champion set
        base_champs = set()
        for m in initial_matches:
            base_champs |= id2champs[m] if m in csv_ids else champs_api(m)

        # Calculate new champion contribution from add_matches
        add_info = []
        for m in add_matches:
            champs = id2champs[m] if m in csv_ids else champs_api(m)
            gain   = len(champs - base_champs)
            add_info.append((gain, m, champs))

        # Sort by contribution (descending)
        add_info.sort(reverse=True)

        # Cut to top_k (optional)
        if top_k is not None and len(add_info) > top_k:
            add_info = add_info[:top_k]

        # Combination search
        best_combo, best_cov, best_cnt = None, -1.0, 0
        total_csv = len(csv_matches)

        for combo in itertools.combinations(add_info, n):
            combo_ids   = [m for _, m, _ in combo]
            combo_champ = base_champs.union(*(c for _, _, c in combo))

            covered = sum(1 for _, champs in csv_matches if champs <= combo_champ)
            cov_pct = covered / total_csv * 100

            if cov_pct > best_cov:
                best_cov, best_combo, best_cnt = cov_pct, combo_ids, covered

        # Output results
        if best_combo is None:
            print("No matching combination found.")
            return None

        print(f"\n Initial {len(initial_matches)} matches + {n} additional matches combination result")
        print(f"  > Best match_id combination: {best_combo}")
        print(f"  > Coverage           : {best_cnt}/{total_csv} "
            f"({best_cov:.2f}%)")
        unique_champs = base_champs.union(*(id2champs[m] if m in csv_ids else champs_api(m) for m in best_combo))
        print(f"  > Unique champion count: {len(unique_champs)}")

        return best_combo, best_cov, best_cnt, total_csv


    def save_summoner_leagueinfo_of_replays(
        self,
        replay_dir: str,
        save_folder: str,
        queue_type: str = "RANKED_SOLO_5x5",
        max_workers: int = 8,
    ):
        """
        Save rank info of 10 summoners as JSON (one file per match)
        in parallel based on .rofl files in replay_dir.
        """
        import os, json, requests

        os.makedirs(save_folder, exist_ok=True)

        rofl_files = [f for f in os.listdir(replay_dir) if f.endswith(".rofl")]
        print(f"Processing {len(rofl_files)} replay files...")

        # Internal function: Process one match
        def process_one_match(rofl_name: str):
            match_id_api = os.path.splitext(rofl_name)[0].replace("-", "_")
            json_path = os.path.join(save_folder, f"{match_id_api}.json")
            if os.path.exists(json_path):
                return  # Already exists

            # 1) match -> 10 puuids
            url_match = (f"https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/"
                        f"{match_id_api}?api_key={self._config.api_key}")
            try:
                match_data = requests.get(url_match, timeout=5).json()
                puuids = match_data.get("metadata", {}).get("participants", [])
                if len(puuids) != 10:
                    return
            except Exception:
                return

            # 2) Each puuid -> rank info
            players = []
            for puuid in puuids:
                url_rank = (f"https://{self._config.region.lower()}.api.riotgames.com/lol/league/v4/entries/by-puuid/"
                            f"{puuid}?api_key={self._config.api_key}")
                try:
                    data = requests.get(url_rank, timeout=5).json()
                except Exception:
                    continue

                solo = next((e for e in data if e.get("queueType") == queue_type), None)
                players.append({
                    "puuid": puuid,
                    "tier":  solo.get("tier", "UNRANKED") if solo else "UNRANKED",
                    "rank":  solo.get("rank", "")         if solo else "",
                    "leaguePoints": solo.get("leaguePoints", 0) if solo else 0,
                    "wins":  solo.get("wins", 0)          if solo else 0,
                    "losses": solo.get("losses", 0)       if solo else 0,
                })

            # 3) Save JSON
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({"match_id": match_id_api, "players": players},
                        jf, ensure_ascii=False, indent=4)

        # Progress bar (tqdm)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_one_match, r): r for r in rofl_files}

            for _ in tqdm(as_completed(futures), total=len(rofl_files), desc="Fetching rank data"):
                pass

    def save_summoner_leagueinfo_from_csv(
        self,
        csv_in: str  = "data/match_champ_dict(25.13).csv",
        save_folder: str = "data/match_info",
        queue_type: str = "RANKED_SOLO_5x5",
        max_workers: int = 4,
    ):
        """
        Save summoner rank info as JSON (one file per match) for each match_id in CSV.
        """
        import pandas as pd, os, json, requests

        # Prepare output folder
        os.makedirs(save_folder, exist_ok=True)

        # Load match_id list
        df = pd.read_csv(csv_in)
        if "match_id" not in df.columns:
            raise ValueError(f"'match_id' column not found in {csv_in}")

        match_ids = [mid for mid in df["match_id"].dropna().unique()
                    if str(mid).startswith(self._config.region_prefix)]

        # Filter out already processed match_ids
        to_process = []
        for mid in match_ids:
            json_path = os.path.join(save_folder, f"{mid}.json")
            if not os.path.exists(json_path):
                to_process.append(mid)

        print(f"Total match_ids: {len(match_ids)}, Already processed: {len(match_ids) - len(to_process)}, To process: {len(to_process)}")

        if not to_process:
            print("All match_ids have already been processed.")
            return

        # Counter for progress tracking
        completed_count = [0]  # Use list to allow modification in nested function
        error_count = [0]

        # Internal function: Process one match
        def process_one_match(match_id_api: str):
            json_path = os.path.join(save_folder, f"{match_id_api}.json")
            if os.path.exists(json_path):        # Skip if already exists
                return "skipped"

            # 1) match -> 10 puuids
            url_match = (f"https://{self._config.routing}.api.riotgames.com/lol/match/v5/matches/"
                        f"{match_id_api}?api_key={self._config.api_key}")
            try:
                resp = requests.get(url_match, timeout=10)
                if resp.status_code == 429:  # Rate limit
                    wait = int(resp.headers.get("Retry-After", 5))
                    time.sleep(wait + 1)
                    resp = requests.get(url_match, timeout=10)
                match_data = resp.json()
                puuids = match_data.get("metadata", {}).get("participants", [])
                if len(puuids) != 10:
                    return f"error: puuids count {len(puuids)}"
            except Exception as e:
                return f"error: match API failed - {e}"

            # 2) Each puuid -> rank info
            players = []
            for puuid in puuids:
                url_rank = (f"https://{self._config.region.lower()}.api.riotgames.com/lol/league/v4/entries/by-puuid/"
                            f"{puuid}?api_key={self._config.api_key}")
                try:
                    resp = requests.get(url_rank, timeout=10)
                    if resp.status_code == 429:  # Rate limit
                        wait = int(resp.headers.get("Retry-After", 5))
                        time.sleep(wait + 1)
                        resp = requests.get(url_rank, timeout=10)
                    data = resp.json()
                except Exception:
                    continue
                solo = next((e for e in data if e.get("queueType") == queue_type), None)
                players.append({
                    "puuid": puuid,
                    "tier":  solo.get("tier", "UNRANKED") if solo else "UNRANKED",
                    "rank":  solo.get("rank", "")         if solo else "",
                    "leaguePoints": solo.get("leaguePoints", 0) if solo else 0,
                    "wins":  solo.get("wins", 0)          if solo else 0,
                    "losses": solo.get("losses", 0)       if solo else 0,
                })

            # 3) Save JSON
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({"match_id": match_id_api, "players": players},
                        jf, ensure_ascii=False, indent=4)
            return "success"

        # Process with ThreadPoolExecutor (simple print progress)
        print(f"Starting to process {len(to_process)} match_ids...")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_one_match, mid): mid for mid in to_process}

            for fut in as_completed(futures):
                mid = futures[fut]
                try:
                    result = fut.result()
                    if result == "success":
                        completed_count[0] += 1
                    elif result and result.startswith("error"):
                        error_count[0] += 1
                        print(f"  [{mid}] {result}")
                except Exception as e:
                    error_count[0] += 1
                    print(f"  [{mid}] Exception: {e}")

                # Print progress every 50 matches
                total_done = completed_count[0] + error_count[0]
                if total_done % 50 == 0 or total_done == len(to_process):
                    print(f"Progress: {total_done}/{len(to_process)} (success: {completed_count[0]}, errors: {error_count[0]})")

        print(f"Completed! Processed: {completed_count[0]}, Errors: {error_count[0]}")



    def save_match_avg_score_from_json(
        self,
        match_info_dir: str,
        output_csv_path: str,
    ):
        """
        From *.json files in match_info_dir (summoner rank info per match) ->
        Calculate 10-player average score and save to output_csv_path.
        """

        # Score conversion criteria
        tier_points = {
            "BRONZE": 0, "SILVER": 400, "GOLD": 800,
            "PLATINUM": 1200, "EMERALD": 1600,
            "DIAMOND": 2000, "MASTER": 2400,
            "GRANDMASTER": 2800, "CHALLENGER": 3200,
        }
        rank_points = {"I": 300, "II": 200, "III": 100, "IV": 0}

        # Prepare result CSV header (create if not exists)
        file_exists = os.path.exists(output_csv_path)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        if not file_exists:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["match_id", "avg_score"])

        # Iterate JSON files
        json_files = [fn for fn in os.listdir(match_info_dir) if fn.endswith(".json")]
        for fn in tqdm(json_files, desc="Calculating average scores"):
            path = os.path.join(match_info_dir, fn)
            with open(path, "r", encoding="utf-8") as jf:
                data = json.load(jf)

            match_id = data.get("match_id")
            players  = data.get("players", [])
            if not match_id or len(players) != 10:
                continue  # Insufficient info -> skip

            scores = []
            for p in players:
                tier = str(p.get("tier", "")).upper()
                rank = str(p.get("rank", "")).upper()
                lp   = int(p.get("leaguePoints", 0))

                base  = tier_points.get(tier, 0)
                bonus = rank_points.get(rank, 0)
                scores.append(base + bonus + lp)

            if not scores:
                continue

            avg_score = sum(scores) / len(scores)

            # Write to CSV (append)
            with open(output_csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([match_id, avg_score])


    # ==========================================================
    # Helper methods for timeline analysis
    # ==========================================================
    @staticmethod
    def _format_timestamp_ms(ts_ms: int) -> str:
        m, s = divmod(round(ts_ms / 1000), 60)
        return f"{m}:{s:02d}"

    @staticmethod
    def _get_champion_kill_logs(tl_data: dict) -> list[dict]:
        logs = []
        for frame in tl_data["info"]["frames"]:
            for ev in frame.get("events", []):
                if ev.get("type") == "CHAMPION_KILL":
                    logs.append({
                        **ev,
                        "formatted_time": RiotAPILegacy._format_timestamp_ms(ev["timestamp"]),
                        "minute": ev["timestamp"] // 60000
                    })
        return logs

    @staticmethod
    def _bind_kill_event_per_minute(kill_logs: list[dict]) -> dict:
        by_min = defaultdict(list)
        for ev in kill_logs:
            by_min[ev["minute"]].append({
                "formatted_time": ev["formatted_time"],
                "position": ev.get("position")
            })
        return {
            str(m): {"event_count": len(v), "events": v}
            for m, v in by_min.items() if len(v) >= 4
        }

    @staticmethod
    def _get_frequent_kill_minute(kill_per_min: dict, kill_events_threshold: int = 4) -> list[int]:
        return sorted(int(m) for m, info in kill_per_min.items()
                      if info["event_count"] >= kill_events_threshold)

    def _process_one_match(self, matchid_num: str, kill_events_threshold) -> list[int]:
        tl = self._get_match_timeline_data(matchid_num)
        kills = self._get_champion_kill_logs(tl)
        per_min = self._bind_kill_event_per_minute(kills)
        return self._get_frequent_kill_minute(per_min, kill_events_threshold)

    # ==========================================================
    # Save .json method (multi-processing)
    # ==========================================================
    def save_kill_events_from_matchlist(
        self,
        matchids: list[str],
        output_json_path: str | None = None,
        kill_events_threshold: int = 4,
        max_workers: int = 2,
    ):
        """
        Receive match_id list, extract minutes where kill events per minute >= threshold
        from timeline, and save cumulatively to output_json_path.
        """

        # Load existing results
        if os.path.exists(output_json_path):
            with open(output_json_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = {"data": {}}

        nums = self.get_matchids2(matchids)
        to_do = [m for m in nums if m not in result["data"]]
        if not to_do:
            print("All match_ids have already been processed.")
            return

        # Multi-processing
        from multiprocessing import set_start_method
        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            fut_map = {exe.submit(self._process_one_match, mid, kill_events_threshold): mid for mid in to_do}

            for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="Processing matches"):
                mid = fut_map[fut]
                try:
                    minutes = fut.result()
                    result["data"][mid] = minutes
                except Exception as e:
                    print(f"{mid} failed: {e}")

                # Intermediate save
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

        print("All matches processed!")
