import os 
import re
### easyorc ###
import easyocr
import cv2
import numpy as np

class OcrCenter(object):

    def __init__(self, project_folder_dir):
        self.project_folder_dir = project_folder_dir

    def get_ocr(self):
        replay_folders = [folder for folder in os.listdir(self.project_folder_dir) if folder[:3] == 'KR-']
        print(replay_folders)
        def extract_number_from_filename(filename):
                """
                Function to extract numbers from filename.
                """
                match = re.search(r'(\d+)_team_kda', filename)
                if match:
                    return int(match.group(1))
                return float('inf')  # Sort to end if no number exists

        def sort_filenames(filenames):
            """
            Function to sort given filename list in numeric order.
            """
            return sorted(filenames, key=extract_number_from_filename)
        
        def read_file(full_path):
            img_array = np.fromfile(full_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        
        def lane_filtering(info_list):

            # Remove suppression gold entries (use list comprehension instead of modifying during iteration)
            info_list = [item for item in info_list if not (item and item[-1] == "G")]

            filtered_team_info = {}
            user_form = [{"K/D/A" : [],"CS" : 0},{"K/D/A" : [],"CS" : 0},{"K/D/A" : [],"CS" : 0},{"K/D/A" : [],"CS" : 0},{"K/D/A" : [],"CS" : 0},{"K/D/A" : [],"CS" : 0},{"K/D/A" : [],"CS" : 0},{"K/D/A" : [],"CS" : 0},{"K/D/A" : [],"CS" : 0},{"K/D/A" : [],"CS" : 0}]
            #print(info_list)
            j = 0
            for i in range(len(info_list)):
            
                if len(info_list[i].split("/")) > 1:
                    user_form[j]['K/D/A'] = info_list[i].split("/")
                # In case of CS
                if len(info_list[i].split("/")) == 1:
                    user_form[j]['CS'] = int(info_list[i])

                # When filled, add to info and reset user_form
                if  len(user_form[j]['K/D/A']) > 1 and user_form[j]['CS'] != 0:
                    #filtered_team_info[j] = user_form)
                    #print(filtered_team_info)
                    #user_form['suppression_gold'] = '0'
                    #user_form['K/D/A'] = []
                    #user_form['CS'] = 0
                    j += 1


            return user_form
                


        for replay_folder in replay_folders:
            # Go to ../All folder to view captured screens from full sight
            replay_folder_allow_all_sight = rf'{self.project_folder_dir}\{replay_folder}\All'
            print(replay_folder_allow_all_sight)
            filenames = [file for file in os.listdir(replay_folder_allow_all_sight) if file[-12:] == 'team_kda.png']
            sorted_filenames = sort_filenames(filenames)
            print(filenames)
            for file in sorted_filenames:
                full_file_path = rf'{replay_folder_allow_all_sight}\{file}'
                reader = easyocr.Reader(['ko'], gpu=True)
                img = read_file(full_file_path)
                info_list = reader.readtext(img, detail=0, allowlist="0123456789/G")
                filtered_info = lane_filtering(info_list)
                print("KDA, CS from this frame:\n", filtered_info)
                



    

    