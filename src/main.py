from classes import Rating, EloRating, Glicko2, FootballData, plot_ratings
import numpy as np

# Loading some data
football = FootballData(2000)
games = football.fetch_data()

# Initialize Glicko2 and EloRating objects
glicko2 = Glicko2()
elo = EloRating()

# Initialize dictionaries to store ratings
ratings_dicts = {glicko2.type: {}, elo.type: {}}

# Copy the games DataFrame
df = games.copy()


# Add columns to store Glicko2 and Elo ratings
df["glicko2_rating"] = np.nan
df["elo_rating"] = np.nan


# The Loop
for row_index, row in df.iterrows():
    team = row["team"]
    opponent = row["opponent"]
    is_winner = row["is_winner"]
    year_week = str(row["season_year"]) + "-" + row["week"]

    for rating in [glicko2, elo]:
        rating_dict = ratings_dicts[rating.type]
        # Update ratings for the team
        if team not in rating_dict:
            rating_dict[team] = rating.create_rating()
        if opponent not in rating_dict:
            rating_dict[opponent] = rating.create_rating()

        # Update the rating based on whether the team won or lost
        if is_winner:
            new_rating_team, new_rating_opponent = rating.rate_1vs1(
                rating_dict[team], rating_dict[opponent]
            )
        else:
            new_rating_opponent, new_rating_team = rating.rate_1vs1(
                rating_dict[opponent], rating_dict[team]
            )

        rating_dict[team] = new_rating_team
        rating_dict[opponent] = new_rating_opponent

        # Add ratings to the DataFrame
        df.loc[row_index, rating.type + "_rating"] = new_rating_team.mu


plot_ratings(df, "Minnesota Vikings")
