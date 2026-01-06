import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np


class ImageEditor(object):

    def __init__(self,
            dataset_dir
            ):
        self.dataset_dir = dataset_dir

    def run_editor(self , team):

        # ファイル構造: dataset > 各リプレイ用フォルダ > black > フレーム(.png)

        raw_folders = os.listdir(self.dataset_dir)
        raw_folders = [folder for folder in raw_folders if folder.startswith('KR')]
        print(raw_folders)
        
        for folder in tqdm(raw_folders, leave=True):
            raw_folder_dir = os.path.join(self.dataset_dir, folder) #
            # os.mkdir(raw_folder_dir)  # ~/gameid

            team_path = os.path.join(raw_folder_dir, team)  
            # os.mkdir(team_path) # ~/gameid/team

            gray_path = os.path.join(team_path, 'black')
            try:
                os.mkdir(gray_path) # ~/gameid/team/black
            except:
                pass
            
            raw_files = os.listdir(team_path)

            raw_files = [raw_file for raw_file in raw_files if raw_file.endswith('.png')]
            for raw_file in raw_files:
                # file: 前処理するファイル名（ディレクトリ含まず）
                img = cv2.imread(rf'{team_path}\{raw_file}', cv2.IMREAD_GRAYSCALE)
                ret, thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
                img_id = raw_file.split('.')[0] 
                cv2.imwrite(rf'{gray_path}\{img_id}.png' , thresh)
               
            time.sleep(0.01)

        

    # folder_dir => 例: C:\Users\username\Desktop\pyLoL\CHALLENGER
    # folder_dirに各チームごとにwanted_frames分だけ画像が残るようにしたい

    def fit_frame_length(self, folder_dir , wanted_frames):
        match_folders = os.listdir(folder_dir)

        # 各試合フォルダ => 例: C:\Users\username\Desktop\pyLoL\CHALLENGER\KR-6415928037
        for m_folder in match_folders:

            # レッド、ブルー
            team_folders = os.listdir(f'{folder_dir}\{m_folder}')

            for t_folder in team_folders:
                initial_size = len(os.listdir(f'{folder_dir}\{m_folder}\{t_folder}'))
                diff = len(os.listdir(f'{folder_dir}\{m_folder}\{t_folder}')) - wanted_frames  # diff: wanted_framesよりどのくらい超過したか

                if diff >= 0:
                    # 超過した数だけ削除する

                    for i in range(diff):

                        # 元の数: X
                        # 目標の数: Y
                        # 減らすべきインデックス
                        # 0, 1, 2, 3, ... , X-2, X-1
                        # 0 ,
                        # print(folder_dir , m_folder , t_folder)
                        removed_index = (initial_size//diff)*i
                       
                        try:
                            os.remove(rf'{folder_dir}\{m_folder}\{t_folder}\black\{removed_index}_minimap.png')
                        except:
                            os.remove(rf'{folder_dir}\{m_folder}\{t_folder}\black\{removed_index+1}_minimap.png')
                        pass
                        
                else:
                    print(m_folder + t_folder , diff)
                