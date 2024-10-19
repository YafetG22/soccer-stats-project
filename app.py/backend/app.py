import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests

st.title('Soccer Player Stats Explorer')

st.markdown("""
This app performs simple web scraping of soccer player stats data!
* **Python libraries:** base64, pandas, streamlit, requests
* **Data source:** [FBref.com](https://fbref.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(2000, 2023))))

@st.cache
def load_data(year):
    url = f"https://fbref.com/en/comps/9/{year}-{year+1}/stats/{year}-{year+1}-Premier-League-Stats"
    html = pd.read_html(requests.get(url).text)
    df = html[0]
    raw = df.drop(df[df['Player'] == 'Player'].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats

playerstats = load_data(selected_year)

player_name_input = st.sidebar.text_input("Search Player", '')
sorted_unique_team = sorted(playerstats['Squad'].unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
unique_pos = playerstats['Pos'].unique()
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)
age_min, age_max = st.sidebar.slider('Age Range', int(playerstats['Age'].min()), int(playerstats['Age'].max()), (18, 35))

if 'Gls' in playerstats.columns:
    goals_min, goals_max = st.sidebar.slider('Goals Scored', int(playerstats['Gls'].min()), int(playerstats['Gls'].max()), (0, 20))

df_selected_team = playerstats[(playerstats['Player'].str.contains(player_name_input)) &
                               (playerstats['Squad'].isin(selected_team)) &
                               (playerstats['Pos'].isin(selected_pos)) &
                               (playerstats['Age'].astype(int) >= age_min) &
                               (playerstats['Age'].astype(int) <= age_max)]

if 'Gls' in playerstats.columns:
    df_selected_team = df_selected_team[(df_selected_team['Gls'].astype(int) >= goals_min) &
                                        (df_selected_team['Gls'].astype(int) <= goals_max)]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="soccer_playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(f)

if st.checkbox('Show Goals vs Assists'):
    if 'Gls' in df_selected_team.columns and 'Ast' in df_selected_team.columns:
        st.subheader('Goals vs Assists')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_selected_team, x='Gls', y='Ast', hue='Squad', s=100, palette='viridis')
        plt.xlabel('Goals')
        plt.ylabel('Assists')
        plt.title('Goals vs Assists by Team')
        st.pyplot()

if st.sidebar.checkbox('Show Top Players'):
    top_by = st.sidebar.selectbox('Sort by:', ['Gls', 'Ast'])
    top_n = st.sidebar.slider('Top N Players', 1, 20, 10)
    df_top_players = df_selected_team.sort_values(by=[top_by], ascending=False).head(top_n)
    st.subheader(f'Top {top_n} Players by {top_by}')
    st.dataframe(df_top_players)

if st.checkbox('Show Age Distribution'):
    st.subheader('Age Distribution of Selected Teams')
    plt.figure(figsize=(10, 6))
    sns.histplot(df_selected_team['Age'].astype(int), bins=15, kde=True, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    st.pyplot()

if st.sidebar.checkbox('Compare Two Players'):
    player_1 = st.sidebar.selectbox('Select Player 1', df_selected_team['Player'].unique())
    player_2 = st.sidebar.selectbox('Select Player 2', df_selected_team['Player'].unique())
    df_compare = df_selected_team[df_selected_team['Player'].isin([player_1, player_2])]
    st.subheader(f'Comparison Between {player_1} and {player_2}')
    st.write(df_compare)

if st.button('Save Filtered Data'):
    df_selected_team.to_csv('filtered_data.csv', index=False)
    st.success('Filtered data saved as filtered_data.csv')
