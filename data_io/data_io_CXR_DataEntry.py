import os
import datetime
from sklearn.preprocessing import LabelEncoder

from data_io.misc_DY import *


"""
 Getting Image Information
"""


def get_img_path_list(dataset_dir, log_text):
    # #### Get image path
    img_path_list = []
    for (Par, Subs, Files) in os.walk(dataset_dir):
        for file in Files:
            ext = os.path.splitext(file)[-1]
            if ext in ['.JPEG', '.JPG', '.PNG', '.jpeg', '.jpg', '.png']:
                img_path = os.path.join(Par, file)
                img_path_list.append(img_path)
    print('>>> img_path_list : ', len(img_path_list))
    print('>>> img_path_list : \r', len(img_path_list), file=log_text)
    print_1by1(img_path_list[0:3])
    print(img_path_list[0:3], file=log_text)
    print('\n'+'-'*70)
    print('\n\r'+'-'*70, file=log_text)

    return img_path_list


"""
 Getting Excel Information FOR DataEntry
"""


def get_data_entry_excel_info(label_file, log_text):
    with open(label_file, mode='r', encoding='utf-8') as txt_file:
        txt_file_lines = txt_file.readlines()
    excel_info = []
    for each_line in txt_file_lines:
        tmp1 = each_line.split(',')  # '\t'
        tmp2 = list(map(lambda s: s.strip(' \n'), tmp1))
        excel_info.append(tmp2)
    main_excel_info = excel_info[1:]
    print('\n')
    print('\n\r', file=log_text)
    print('>>> main_excel_info: ', len(main_excel_info))
    print('>>> main_excel_info: ', len(main_excel_info), '\r', file=log_text)
    print_1by1(main_excel_info[:3])
    print(main_excel_info[:3], file=log_text)
    print('\n')
    print('\n\r', file=log_text)

    # #### Extract the distinct info that only includes 'Target Findings'.
    # 질환없음 #경화 #부종/수종 #(폐)기종 #섬유증 #탈장 #늑막두꺼워짐
    NOT_IN = ['|', 'No Finding', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Hernia', 'Pleural_Thickening']
    distinct_excel_info = [line for i, line in enumerate(main_excel_info)
                           if not any(st in line[1] for st in NOT_IN)]
    print('>>> distinct_excel_info: ', len(distinct_excel_info))
    print('>>> distinct_excel_info: ', len(distinct_excel_info), '\r', file=log_text)
    print_1by1(distinct_excel_info[:3])
    print(distinct_excel_info[:3], file=log_text)
    print('\n')
    print('\n\r', file=log_text)

    # #### Convert labels from strings into binary with scikit.preprocessing.
    mlb = LabelEncoder()
    labels = [line[1] for line in distinct_excel_info]
    labels_ = mlb.fit_transform(labels)
    print('>>> labels_: ', len(labels_))
    print('>>> labels_: ', len(labels_), '\r', file=log_text)

    file_info_list_ = []
    for i, line in enumerate(distinct_excel_info):
        # ['Image Index', 'Finding Labels', 'OriginalImage[W','H]', 'OriginalImagePixelSpacing[x','y]']
        # ['Follow-up #','Patient ID'], 'Patient Age', 'Patient Gender', 'View Position',
     file_info_list_.append([line[0], labels_[i], line[2], line[3], line[4], line[5], line[6], line[7:9], line[9:]])

    print('>>> file_info: ', len(file_info_list_))
    print('>>> file_info: ', len(file_info_list_), '\r', file=log_text)
    print_1by1(file_info_list_[:3])
    print(file_info_list_[:3], file=log_text)
    print(excel_info[0], '\n')
    print(excel_info[0], '\n\r', file=log_text)
    print('\r'+'-'*70)
    print('\n\r'+'-'*70, file=log_text)

    return file_info_list_


"""
 Matching Image path and Patient Info.
"""


def match_pathNinfo(file_info_list, img_path_list, log_text):
    time_start = datetime.datetime.now()
    print("     @ Matching starts at " + str(time_start))
    print("     @ Matching starts at " + str(time_start), '\r', file=log_text)

    matching_list = []
    for idx, value in enumerate(file_info_list):
        file_name = value[0]
        label = value[1]
        pati_info = value[2]
        for each_dir in img_path_list:  # 106116
            if file_name in each_dir:
                matching_list.append([file_name, label, pati_info, each_dir])
                break
            else:
                pass

    img_name_list = [val[0] for val in matching_list]
    img_dir_list = [val[-1] for val in matching_list]
    label_list = [val[1] for val in matching_list]

    print('\n')
    print('\n\r', file=log_text)
    print('>>> matching_list : ', len(matching_list))
    print('>>> matching_list : ', len(matching_list), '\r', file=log_text)
    print_1by1(matching_list[:3])
    print(matching_list[:3], file=log_text)
    print('\n')
    print('\n\r', file=log_text)
    print("     @ DONE at " + str(datetime.datetime.now()) + '\r')
    print("     @ DONE at " + str(datetime.datetime.now()) + '\r', file=log_text)
    print("     @ This took for %s" % str(datetime.datetime.now() - time_start))
    print("     @ This took for %s" % str(datetime.datetime.now() - time_start), '\r', file=log_text)
    print('\r'+'='*100)
    print('\r'+'='*100, file=log_text)

    return img_name_list, img_dir_list, label_list


if __name__ == "__main__":
    get_img_path_list()
    get_data_entry_excel_info()
    match_pathNinfo()
