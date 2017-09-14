import pandas as pd
import numpy as np
import sqlite3
import math

# set database path
database_path = "../../data/database.sqlite"

# helper functions
def get_rating(player_id):
    if math.isnan(player_id):
        return np.nan
    return player_lvl['overall_rating'][player_id]

def get_outcome(home, away):
    if home > away:
        return 0
    elif home == away:
        return 1
    else:
        return 2

# open connection with database and get relevant data
with sqlite3.connect(database_path) as con:
    sql = '''
    SELECT *
    FROM match 
    '''
    df_match = pd.read_sql_query(sql, con)
    
with sqlite3.connect(database_path) as con:
    sql = '''
    SELECT player_api_id, overall_rating
    FROM player_attributes
    '''
    player_lvl = pd.read_sql_query(sql, con)

# select relevant variables
df = df_match[['stage', 'home_team_goal', 'away_team_goal','B365H','B365D','B365A','BWH',
               'BWD','BWA','IWH','IWD','IWA','LBH','LBD','LBA','WHH','WHD','WHA','SJH','SJD','SJA',
               'VCH','VCD','VCA','GBH','GBD','GBA']]
df_lvls = df_match[['home_player_1','home_player_2','home_player_3','home_player_4','home_player_5',
                   'home_player_6','home_player_7','home_player_8','home_player_9','home_player_10',
                   'home_player_11','away_player_1','away_player_2','away_player_3','away_player_4',
                   'away_player_5','away_player_6','away_player_7','away_player_8','away_player_9',
                   'away_player_10','away_player_11']]

# get player skill-levels
player_lvl = player_lvl.groupby('player_api_id').mean().to_dict()
df = pd.concat([df, df_lvls.applymap(lambda x: get_rating(x))], axis=1)

# get target variable
df['target'] = df.apply(lambda x: get_outcome(x['home_team_goal'], x['away_team_goal']), axis=1)
df = df.drop(['home_team_goal', 'away_team_goal'], axis=1)

# drop nans
df.dropna(inplace=True)

# save processed data
df.to_csv('../../data/processed.csv', index=False)
