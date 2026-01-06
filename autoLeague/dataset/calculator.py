import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import requests
from dotenv import load_dotenv

# Load API Key from RiotAPI.env
load_dotenv("RiotAPI.env")
API_KEY = os.environ.get("API_KEY")


class AreaCalculator(object):

    def __init__(self,
            project_folder_dir, tier
            ):
        self.project_folder_dir = project_folder_dir
        self.tier = tier

    @staticmethod
    def get_triangle_roi(image, points):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        roi = cv2.bitwise_and(image, mask)
        return roi

    @staticmethod
    def calculate_pixel_sum(image_gray):
        pixel_sum = np.sum(image_gray)
        return pixel_sum//255


    blue_jug_near_baron = [(370,400),(200,300),(135,165),(100,200),(100,550),(150,550),(200,575)]

    blue_jug_near_drake = [(840-415,830-365),(840-350,830-325),(840-300,830-200),(840-210,830-210),(840-200,830-110),(840-230,830-100),(840-550,830-100),(840-550,830-150),(840-575,830-200),(840-415,830-365)]

    red_jug_near_baron = [(415,365),(350,325),(300,200),(210,210),(200,110),(230,100),(550,100),(550,150),(575,200),(415,365)]

    red_jug_near_drake = [(840-415,830-365),(840-350,830-325),(840-300,830-200),(840-210,830-210),(840-200,830-110),(840-230,830-100),(840-550,830-100),(840-550,830-150),(840-575,830-200),(840-415,830-365)]

    river_near_baron = [(370,400),(200,300),(135,165),(200,110),(210,210),(300,200),(350,325),(415,365),(370,400)]

    river_near_drake = [(840-370,830-400),(840-200,830-300),(840-135,830-165),(840-200,830-110),(840-210,830-210),(840-300,830-200),(840-350,830-325),(840-415,830-365),(840-370,830-400)]


    @staticmethod
    def get_is_win(matchid, team_color):
        matchID = matchid.replace('-','_')
        if team_color == 'Blue':
            return 1 if requests.get(f'https://asia.api.riotgames.com/lol/match/v5/matches/{matchID}?api_key={API_KEY}').json()['info']['teams'][0]['win'] == True else 0
        else:
            return 1 if requests.get(f'https://asia.api.riotgames.com/lol/match/v5/matches/{matchID}?api_key={API_KEY}').json()['info']['teams'][1]['win'] == True else 0
    

    # Read image files

    def get_each_Vision_Area(self,image_dir):    #image_dir example: r'C:\dataset\CHALLENGER\KR-6415928037\Blue\black\105.png'

        #image = cv2.imread(r'C:\dataset\CHALLENGER\KR-6415928037\Blue\black\105.png')
        image = cv2.imread(image_dir)

        triangle_BJ_B = self.get_triangle_roi(image, np.array(self.blue_jug_near_baron))
        area_BJ_B = self.calculate_pixel_sum(triangle_BJ_B)
        #print("Blue team vision area near blue:", area_BJ_B)

        triangle_BJ_D = self.get_triangle_roi(image, np.array(self.blue_jug_near_drake))
        area_BJ_D = self.calculate_pixel_sum(triangle_BJ_D)
        #print("Blue team vision area near red:", area_BJ_D)

        triangle_RJ_B = self.get_triangle_roi(image, np.array(self.red_jug_near_baron))
        area_RJ_B = self.calculate_pixel_sum(triangle_RJ_B)
        #print("Red team vision area near red:", area_RJ_B)

        triangle_RJ_D = self.get_triangle_roi(image, np.array(self.red_jug_near_drake))
        area_RJ_D = self.calculate_pixel_sum(triangle_RJ_D)
        #print("Red team vision area near blue:", area_RJ_D)

        triangle_BARON_RIVER = self.get_triangle_roi(image, np.array(self.river_near_baron))
        area_BARON_RIVER = self.calculate_pixel_sum(triangle_BARON_RIVER)
        #print("Baron riverside vision area:", area_BARON_RIVER)

        triangle_DRAKE_RIVER = self.get_triangle_roi(image, np.array(self.river_near_drake))
        area_DRAKE_RIVER = self.calculate_pixel_sum(triangle_DRAKE_RIVER)
        #print("Dragon riverside vision area:", area_DRAKE_RIVER)

        return [area_BJ_B, area_BJ_D, area_RJ_B, area_RJ_D, area_BARON_RIVER, area_DRAKE_RIVER]

    # Organize vision area info for matches in this tier into CSV file
    def get_each_Vision_Area_Per_Tier(self, project_folder_dir, tier):

        # EXAMPLE ##########################################
        # project_folder_dir : C:\Users\username\Desktop\pyLoL or C:\dataset
        # tier : CHALLENGER
        ####################################################
        INITIAL = 0
        match_folders = os.listdir(f'{project_folder_dir}\{tier}')
        project_folder_dir = f'{project_folder_dir}\{tier}'

        # Each match folder => example: C:\Users\username\Desktop\pyLoL\CHALLENGER\KR-6415928037

        f = open(rf'C:\dataset\{tier}_dataset.csv','a', newline='')
        features = ['matchID']
        for i in range(379):
            features.extend([f'blue_baron{i}',f'blue_dragon{i}',f'red_baron{i}',f'red_dragon{i}',f'baron{i}',f'dragon{i}'])

        features.append('is_win')

        print(len(features))

        wr = csv.writer(f)
        wr.writerow(features)
        f.close()
        for m_folder in tqdm(match_folders):

            # Red, Blue
            team_folders = os.listdir(f'{project_folder_dir}\{m_folder}')

            #os.remove(rf'{folder_dir}\{m_folder}\{t_folder}\black\{removed_index}.png')

            i = INITIAL
            
            for t_folder in team_folders:
                
                data = [f'{m_folder}_{t_folder}']

                #csv file write
                f = open(rf'C:\dataset\{tier}_dataset.csv','a', newline='')
                wr = csv.writer(f)

                files = os.listdir(f'{project_folder_dir}\{m_folder}\{t_folder}\\black')
                for file in files:
                    data.extend(self.get_each_Vision_Area(self,rf'{project_folder_dir}\{m_folder}\{t_folder}\\black\{file}'))

                data.extend(str(self.get_is_win(m_folder, t_folder)))
                wr.writerow(data)
                f.close()
                
                #os.remove(rf'{folder_dir}\{m_folder}\{t_folder}\black\{removed_index}.png')

        print('Done!')