import streamlit as st
import pandas as pd
import altair as alt
from shannon_random import create_dataframes_from_list_play, generate_tree_column_name, \
    make_combinatory_probability_df, generate_prediction


def add_key_to_session(key_name: str, value_name):
    if key_name not in st.session_state:
        st.session_state[key_name] = value_name
    pass


def increment_session_key_name(key_name: str):
    st.session_state[key_name] += 1
    pass


def set_difficulty_depth():
    st.session_state['difficulty_depth'] = st.session_state.difficulty_slider
    pass


def create_score_dataframe():
    human_score = st.session_state['human_score']
    ai_score = st.session_state['ai_score']
    scorer_dataframe = pd.DataFrame.from_dict({'player': ['PLAYER', 'AI'],
                                               'score': [human_score, ai_score]})
    return scorer_dataframe


def create_session_dataframe_from_play(search_depth: int = 2):
    list_human_play = st.session_state['human_play']
    train_df, x_test = create_dataframes_from_list_play(list_human_play, search_depth)
    return train_df, x_test


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
    train_df, x_test = create_session_dataframe_from_play(TREE_DEPTH)
    y_pred = generate_prediction(TREE_COMBINATORY_DEPTH, train_df, x_test, TREE_COLUMN_NAME)
    increment_correct_score(0, y_pred)
    st.session_state['human_play'].append(0)
    pass


def make_blue():
    train_df, x_test = create_session_dataframe_from_play(TREE_DEPTH)
    y_pred = generate_prediction(TREE_COMBINATORY_DEPTH, train_df, x_test, TREE_COLUMN_NAME)
    increment_correct_score(1, y_pred)
    st.session_state['human_play'].append(1)
    pass


st.set_page_config(page_title="Mind Reader",
                   page_icon="🧠")
DEFAULT_DEPTH = 3

add_key_to_session('human_score', 0)
add_key_to_session('ai_score', 0)
add_key_to_session('human_play', [])
add_key_to_session('bot_message', '')
add_key_to_session('difficulty_depth', DEFAULT_DEPTH)

st.subheader('Do you believe that human can produce randomness?')
st.markdown('The objective of this experience is to show that __humans cannot '
            'produce randomness__ and that they end up **repeating patterns**.')
st.markdown('To make it efficient, a high number of turn should be played (250).')
st.write('Please pick a button and the bot will try to guess which one you pick.')

TREE_DEPTH = st.session_state.difficulty_depth
TREE_COMBINATORY_DEPTH = TREE_DEPTH + 1
TREE_COLUMN_NAME = generate_tree_column_name(TREE_COMBINATORY_DEPTH)

all_columns = st.columns([1, 1, 3, 3, 3])

all_columns[0].button('Red', on_click=make_red)
all_columns[1].button('Blue', on_click=make_blue)

bot_message = st.session_state['bot_message']
st.text(bot_message)

score_dataframe = create_score_dataframe()

bar_chart = alt.Chart(score_dataframe).mark_bar().encode(
    x='player',
    y='score',
    color='player'
).properties(width=400)

st.altair_chart(bar_chart)

if len(st.session_state.human_play) > 0:
    st.write(f'Number of play : {len(st.session_state.human_play)}')

with st.expander('Sequence depth explanations'):
    st.markdown('The `sequence depth` parameter lets you define the **length of the sequence** the bot should observe '
                'on all your play to predict your future **action**.')
    st.markdown('For example, if `[Red, Red, Blue, Red]` is played, a `sequence depth` of `2` will result in :')
    example_dataframe = create_dataframes_from_list_play([0, 0, 1, 0], search_depth=2)
    st.dataframe(example_dataframe[0].replace(to_replace={0: 'Red', 1: 'Blue'}))
    st.markdown('Now, if you play `[Red, Blue]` we can try to predict what you **next action** will be.')

st.slider('Sequence Depth',
          min_value=1,
          max_value=10,
          value=DEFAULT_DEPTH,
          step=1,
          on_change=set_difficulty_depth,
          key='difficulty_slider')

with st.expander('Combinatory Probabilities'):
    train_dataframe, x_test_ = create_session_dataframe_from_play(TREE_DEPTH)
    probability_df = make_combinatory_probability_df(TREE_COMBINATORY_DEPTH, train_dataframe, TREE_COLUMN_NAME)
    combinatory_str_df = probability_df.loc[:, probability_df.columns != 'probability']
    combinatory_str_df = combinatory_str_df.replace(to_replace={0: 'Red', 1: 'Blue'})
    combinatory_str_df['probability'] = probability_df['probability']
    st.dataframe(combinatory_str_df.sort_values(by='probability', ascending=False))

st.write("Experience inspired by [Shannon\'s Mind-Reading Machine]"
         "(https://this1that1whatever.com/miscellany/mind-reader/Shannon-Mind-Reading.pdf)")


