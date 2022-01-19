import pandas as pd
import random
import itertools
from app_config import APP_BASEDIR

if '/game' in APP_BASEDIR:
    APP_BASEDIR = APP_BASEDIR.split('/game', 1)[0]


def create_dataframes_from_list_play(list_of_play: list, search_depth: int):
    sliced_ai_play = []
    # Check if we have enough data to create dataset with depth
    if len(list_of_play) - 1 < search_depth:
        # If not, create a random list of number
        random_init_list = [random.randint(0, 2) for i in range(search_depth + 1)]
        list_of_play = random_init_list

    test_list = list_of_play[len(list_of_play) - search_depth: len(list_of_play)]

    for i in range(len(list_of_play)):
        sliced_list = list_of_play[0 + i: search_depth + i + 1]
        if len(sliced_list) == search_depth + 1:
            sliced_ai_play.append(sliced_list)

    df_columns = [f'state_{i + 1}' for i in range(search_depth)] + ['action']
    train_df = pd.DataFrame(sliced_ai_play, columns=df_columns)
    X_test = pd.DataFrame([test_list], columns=df_columns[:-1])

    return train_df, X_test


def generate_tree_column_name(combinatory_depth: int) -> list:
    column_name = ['root']
    for layer in range(1, combinatory_depth):
        column_name.append(f'node_{layer}')
    return column_name


def make_combinatory_probability_df(combinatory_depth: int, train_df: pd.DataFrame, column_name: list) -> pd.DataFrame:
    train_df.columns = column_name
    probability_df = pd.DataFrame(index=range(2 ** combinatory_depth), columns=column_name)

    for index, element in enumerate(map(list, itertools.product([0, 1], repeat=combinatory_depth))):
        probability_df.iloc[index] = element

    probability_value = []
    for index in range(len(probability_df)):
        num_value = (probability_df.iloc[index].values == train_df).all(axis=1).astype(int).sum()
        probability_value.append(num_value * 100 / len(train_df))

    probability_df['probability'] = probability_value

    return probability_df


def generate_prediction(combinatory_depth: int, train_df: pd.DataFrame, X_test: pd.DataFrame, column_name: list) -> int:
    X_test.columns = column_name[:-1]
    probability_df = make_combinatory_probability_df(combinatory_depth, train_df, column_name)
    y_test_probability_df = probability_df.loc[(probability_df[X_test.columns] == X_test.values).all(axis=1)]
    try:
        y_predict = y_test_probability_df[column_name[-1]][y_test_probability_df['probability'].idxmax()]
    except ValueError:
        y_predict = random.randint(0, 1)
    return y_predict
