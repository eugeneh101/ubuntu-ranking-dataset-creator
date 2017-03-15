### written in python 2

"""
This code mimics the creation of .tsv files of the 
prepare_data_maybe_download() function in create_ubuntu_dataset.py from 
https://github.com/rkadlec/ubuntu-ranking-dataset-creator
For this python script, instead of downloading a dataset, pass in a .csv file 
which has your product line's conversations (which has the following columns: 
repid, start_time, text, thread_id, time). Also, specify an output file name 
which the script will save the conversations sorted by thread_ids with the
earliest start_time.
In addition, this code will also generate trainfiles.csv, valfiles.csv, and 
testfiles.csv inside a folder named meta--these .csv files will tell 
create_ubuntu_dataset.py how to create the train.csv, valid.csv, and test.csv, 
which are the actual files that the LSTM needs to train itself.



For example: python prepare_data.py refrigerator_data.csv refrigerator_data_sorted.csv
"""


import sys
import os
import pandas as pd
from datetime import datetime
import itertools
import copy
import glob
import random

def get_time(time_string):
    """From a time stamp string, convert to a datetime object."""
    return datetime.strptime(time_string.split('+')[0], '%Y-%m-%dT%H:%M:%S')

def convert_chat_to_tsv(chat_df, folder_name, index, thread_id):
    if len(list(itertools.groupby(chat_df['repid']))) <= 3:
        return
    chat_df.reset_index(drop=True, inplace=True)
    chat_df['thread_id'] = ([''] + list(chat_df['repid'][:-1]))
    chat_df.columns = ['time', 'speaker', 'text', 'listener']
    chat_df = chat_df.reindex(columns=['time', 'speaker', 'listener', 'text'])
    chat_df.to_csv(folder_name + '/' + index + '_' + thread_id + '.tsv', 
                   sep='\t', header=False, index=False)
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please type in your product line .csv file and what you " + 
              "want to name the product line .csv file sorted by the " +
              "thread_id with the earliest start times.\nFor example: " +
              "python prepare_data.py refrigerator_data.csv " +
              "refrigerator_data_sorted.csv") 
        sys.exit()

    df_refrigerator = pd.read_csv(sys.argv[1])
    print('DataFrame loaded; next step will take about 1 minute')
    
    thread_id_start_time_dict = dict(itertools.izip(df_refrigerator['thread_id'], 
        df_refrigerator['start_time'].apply(get_time)))

    thread_id_ordered = sorted(thread_id_start_time_dict, key=
                               thread_id_start_time_dict.get)
    thread_id_ordered_dict = dict((thread_id, i) for i, thread_id in enumerate(
        thread_id_ordered))
    chat_data = [None] * len(thread_id_ordered_dict)

    for chat_df in df_refrigerator[['time', 'repid', 'text', 'thread_id']
                                  ].groupby('thread_id'):
        chat_data[thread_id_ordered_dict[chat_df[0]]] = chat_df[1]
        
    df_temp = pd.concat(chat_data)
    df_temp.to_csv(sys.argv[2], index=False)
    print('DataFrame sorted by earliest thread_ids saved to .csv file')

    print('Beginning to save individual chats to individual .tsv files')
    print('This can take several minutes')

    if not os.path.isdir('dialogs'):
        os.mkdir('dialogs')
    os.chdir('dialogs')

    folder_name = -1
    for index, (thread_id, chat_df) in enumerate(zip(thread_id_ordered, 
                                                     chat_data)):
        if index % 1000 == 0:
            print("file {} completed".format(index))
            folder_name += 1
            if not os.path.isdir(str(folder_name)):
                os.mkdir(str(folder_name))
        chat_df = copy.deepcopy(chat_df)
        convert_chat_to_tsv(chat_df=chat_df, folder_name=str(folder_name), 
                            index=str(index), thread_id=thread_id)
    print("All tsv files created")
    
    print("Generating the train, validation, and test files in 'meta' folder")
    all_tsv_file_names = pd.Series(glob.glob('*/*'))
    random.shuffle(all_tsv_file_names, random=random.Random(42).random)

    folder_name, file_name = zip(*all_tsv_file_names.apply(lambda file_name: 
        file_name.split('/')))
    all_tsv_file_names = pd.DataFrame({'file_name': file_name, 'folder_name': 
                                       folder_name})

    os.chdir('..')
    if not os.path.isdir('meta'):
        os.mkdir('meta')

    all_tsv_file_names[:int(len(all_tsv_file_names) * 0.7)].to_csv(
        'meta/trainfiles.csv', header=False, index=False)
    all_tsv_file_names[int(len(all_tsv_file_names) * 0.7):int(len(
        all_tsv_file_names) * 0.85)].to_csv(
        'meta/valfiles.csv', header=False, index=False)
    all_tsv_file_names[int(len(all_tsv_file_names) * 0.85):].to_csv(
        'meta/testfiles.csv', header=False, index=False)        
                

    print("All processes done!")