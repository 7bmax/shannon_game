import streamlit as st
import pandas as pd
import altair as alt
from shannon_random import create_dataframes_from_list_play, upload_train_dataset
from make_model import train_and_save_new_model
import keras
import tensorflow as tf
import random
import pickle

def add_key_to_session(key_name: str, value_name):
    if key_name not in st.session_state:
        st.session_state[key_name] = value_name
    pass

def increment_session_key_name(key_name: str):
    st.session_state[key_name] += 1
    pass

def create_score_dataframe():
    human_score = st.session_state['human_score']
    ai_score = st.session_state['ai_score']
    score_dataframe = pd.DataFrame.from_dict({'player': ['human', 'ai'], 
                                              'score': [human_score, ai_score]})
    return score_dataframe

def create_session_dataframe_from_play(search_depth: int=2):
    list_human_play = st.session_state['human_play']
    train_df, X_test = create_dataframes_from_list_play(list_human_play, search_depth)
    return train_df, X_test

# @st.cache(allow_output_mutation=True)
def load_model_from_cache():
    model = keras.models.load_model('/home/maxime/Documents/shannon_game/models/tensor_model')
    return model

def load_sklearn_model():
    filename = 'random_forest_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def predict_value_from_model():
    # model = load_model_from_cache()
    model = load_sklearn_model()
    y_test = model.predict(st.session_state['X_test'])
    st.write(st.session_state['X_test'])
    st.write(y_test)
    return y_test

def increment_correct_score(human_play: int, y_pred: int):
    if human_play == y_pred:
        increment_session_key_name('ai_score')
        if human_play == 0:
            st.session_state['bot_message'] = 'I guessed Red.'
        elif human_play == 1:
            st.session_state['bot_message'] = 'I guessed Blue.'
    else:
        increment_session_key_name('human_score')
        st.session_state['bot_message'] = 'I failed to guess...'
    pass

def make_red():
    y_pred = predict_value_from_model()
    increment_correct_score(0, y_pred)
    st.session_state['human_play'].append(0)
    pass

def make_blue():
    y_pred = predict_value_from_model()
    increment_correct_score(1, y_pred)
    st.session_state['human_play'].append(1)
    pass

def make_random_red():
    y_pred = random.randint(0,1)
    increment_correct_score(0, y_pred)
    st.session_state['human_play'].append(0)
    pass

def make_random_blue():
    y_pred = random.randint(0,1)
    increment_correct_score(1, y_pred)
    st.session_state['human_play'].append(1)
    pass

add_key_to_session('human_score', 0)
add_key_to_session('ai_score', 0)
add_key_to_session('human_play', [])
add_key_to_session('bot_message', '')

st.write('Hello, I will try to guess which button you pick.')

what_todo = st.radio(
    "What do you want to do",
    ('Generate Training Data', 'Play'))

st.write("Let\'s play for real.")    
all_columns = st.columns([1,1,3,3,3,3])

train_df, X_test = create_session_dataframe_from_play(3)

add_key_to_session('X_test', X_test)

upload_train_dataset(train_df)
train_and_save_new_model()

all_columns[0].button('Red', on_click=make_red)
all_columns[1].button('Blue', on_click=make_blue)

bot_message = st.session_state['bot_message']
st.write(bot_message)

if what_todo == 'Generate Training Data':
    st.write('Let\'s play 500 round randomly.')
    
    st.progress(len(st.session_state.human_play) / 500)
    all_columns = st.columns([1,1,3,3,3,3])

    train_df, X_test = create_session_dataframe_from_play(3)

    add_key_to_session('X_test', X_test)

    # upload_train_dataset(train_df)
    # train_and_save_new_model()

    all_columns[0].button('Red', on_click=make_random_red)
    all_columns[1].button('Blue', on_click=make_random_blue)

    bot_message = st.session_state['bot_message']
    st.write(bot_message)

    score_dataframe = create_score_dataframe()

    bar_chart = alt.Chart(score_dataframe).mark_bar().encode(
        x='player',
        y='score', 
        color='player'
    ).properties(width=400)

    st.altair_chart(bar_chart)

    # with st.container():
    #     debug_columns = st.columns(2)
    #     st.write('score_dataframe')
    #     st.dataframe(score_dataframe)
    #     with debug_columns[0]:
    #         st.write('train_df')
    #         st.dataframe(train_df)
    #     with debug_columns[1]:
    #         st.write('X_test')
    #         st.dataframe(X_test)
else:
    st.write("Let\'s play for real.")    
    # all_columns = st.columns([1,1,3,3,3,3])

    # train_df, X_test = create_session_dataframe_from_play(3)

    # add_key_to_session('X_test', X_test)

    # upload_train_dataset(train_df)
    # train_and_save_new_model()

    # all_columns[0].button('Red', on_click=make_red)
    # all_columns[1].button('Blue', on_click=make_blue)

    # bot_message = st.session_state['bot_message']
    # st.write(bot_message)

    # score_dataframe = create_score_dataframe()

    # bar_chart = alt.Chart(score_dataframe).mark_bar().encode(
    #     x='player',
    #     y='score', 
    #     color='player'
    # ).properties(width=400)

    # st.altair_chart(bar_chart)

    # with st.container():
    #     debug_columns = st.columns(2)
    #     st.write('score_dataframe')
    #     st.dataframe(score_dataframe)
    #     with debug_columns[0]:
    #         st.write('train_df')
    #         st.dataframe(train_df)
    #     with debug_columns[1]:
    #         st.write('X_test')
    #         st.dataframe(X_test)

