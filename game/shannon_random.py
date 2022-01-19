import pandas as pd
import random


def create_dataframes_from_list_play(list_of_play: list, search_depth: int):
    sliced_ai_play = []
    # Check if we have enough data to create dataset with depth
    if len(list_of_play) - 1 < search_depth:
        # If not, create a random list of number
        random_init_list = [random.randint(0,2) for i in range(search_depth + 1)]
        list_of_play = random_init_list
    
    test_list = list_of_play[len(list_of_play) - search_depth : len(list_of_play)]
    
    for i in range(len(list_of_play)):
        sliced_list = list_of_play[0 + i : search_depth + i + 1]
        if len(sliced_list) == search_depth + 1:
            sliced_ai_play.append(sliced_list)

    df_columns = [f'state_{i + 1}' for i in range(search_depth)] + ['action']
    train_df = pd.DataFrame(sliced_ai_play, columns = df_columns)
    X_test = pd.DataFrame([test_list], columns = df_columns[:-1])
    
    return train_df, X_test

def upload_train_dataset(train_dataset):
    train_dataset.to_csv("/home/maxime/Documents/shannon_game/data/train_dataset.csv", index=False)