import pandas as pd
import datetime
import numpy as np
import math
from tqdm import tqdm
import logging
import functools
import itertools
import time
from typing import Optional, List, Tuple
import os

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#: The actual score for win
WIN = 1.0
#: The actual score for draw
DRAW = 0.5
#: The actual score for loss
LOSS = 0.0

K_FACTOR = 12


MU = 1500
SIGMA = 0.06
#: A constant which is used to standardize the logistic function to
#: `1/(1+exp(-x))` from `1/(1+10^(-r/400))`
Q = math.log(10) / 400


# Glicko related
VOLATILITY = 0.06
TAU = 0.2
EPSILON = 0.000001
DELTA = 0.0001
PHI = 350


def utctime():
    """A function like :func:`time.time` but it uses a time of UTC."""
    return time.mktime(datetime.datetime.utcnow().timetuple())


class Rating(object):
    def __init__(self, mu=MU, phi=PHI, sigma=SIGMA, type=""):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.type = type

    def __repr__(self):
        c = type(self)
        args = (self.type, c.__name__, self.mu, self.phi, self.sigma)
        return "%s.%s(mu=%.3f, phi=%.3f, sigma=%.3f)" % args

    def rate(self):
        raise NotImplemented

    def rate_1vs1(self):
        raise NotImplemented


class EloRating(object):
    def __init__(self, mu=MU, k_factor=K_FACTOR):
        self.k_factor = k_factor
        self.type = "elo"
        self.mu = mu

    def create_rating(self, mu=None):
        if mu is None:
            mu = self.mu
        return Rating(mu, type=self.type)

    def rate(self, rating1, rating2, outcome):
        expected_score = self._calculate_expected_score(rating1, rating2)
        delta = self.k_factor * (outcome - expected_score)
        new_mu = rating1.mu + delta
        return self.create_rating(mu=new_mu)

    def _calculate_expected_score(self, rating1, rating2):
        return 1 / (1 + math.pow(10, (rating1.mu - rating2.mu) / 400))

    def rate_1vs1(self, rating1, rating2, drawn=False):
        outcome = DRAW if drawn else WIN
        new_rating1 = self.rate(rating1, rating2, outcome)
        new_rating2 = self.rate(rating2, rating1, 1 - outcome)  # Opposite outcome
        return new_rating1, new_rating2


class Glicko2(object):
    def __init__(self, mu=MU, phi=PHI, sigma=SIGMA, tau=TAU, epsilon=EPSILON):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.tau = tau
        self.epsilon = epsilon
        self.type = "glicko2"

    def create_rating(self, mu=None, phi=None, sigma=None):
        if mu is None:
            mu = self.mu
        if phi is None:
            phi = self.phi
        if sigma is None:
            sigma = self.sigma
        return Rating(mu, phi, sigma, type=self.type)

    def scale_down(self, rating, ratio=173.7178):
        mu = (rating.mu - self.mu) / ratio
        phi = rating.phi / ratio
        return self.create_rating(mu, phi, rating.sigma)

    def scale_up(self, rating, ratio=173.7178):
        mu = rating.mu * ratio + self.mu
        phi = rating.phi * ratio
        return self.create_rating(mu, phi, rating.sigma)

    def reduce_impact(self, rating):
        """The original form is `g(RD)`. This function reduces the impact of
        games as a function of an opponent's RD.
        """
        return 1.0 / math.sqrt(1 + (3 * rating.phi**2) / (math.pi**2))

    def expect_score(self, rating, other_rating, impact):
        return 1.0 / (1 + math.exp(-impact * (rating.mu - other_rating.mu)))

    def determine_sigma(self, rating, difference, variance):
        """Determines new sigma."""
        phi = rating.phi
        difference_squared = difference**2
        # 1. Let a = ln(s^2), and define f(x)
        alpha = math.log(rating.sigma**2)

        def f(x):
            """This function is twice the conditional log-posterior density of
            phi, and is the optimality criterion.
            """
            tmp = phi**2 + variance + math.exp(x)
            a = math.exp(x) * (difference_squared - tmp) / (2 * tmp**2)
            b = (x - alpha) / (self.tau**2)
            return a - b

        # 2. Set the initial values of the iterative algorithm.
        a = alpha
        if difference_squared > phi**2 + variance:
            b = math.log(difference_squared - phi**2 - variance)
        else:
            k = 1
            while f(alpha - k * math.sqrt(self.tau**2)) < 0:
                k += 1
            b = alpha - k * math.sqrt(self.tau**2)
        # 3. Let fA = f(A) and f(B) = f(B)
        f_a, f_b = f(a), f(b)
        # 4. While |B-A| > e, carry out the following steps.
        # (a) Let C = A + (A - B)fA / (fB-fA), and let fC = f(C).
        # (b) If fCfB < 0, then set A <- B and fA <- fB; otherwise, just set
        #     fA <- fA/2.
        # (c) Set B <- C and fB <- fC.
        # (d) Stop if |B-A| <= e. Repeat the above three steps otherwise.
        while abs(b - a) > self.epsilon:
            c = a + (a - b) * f_a / (f_b - f_a)
            f_c = f(c)
            if f_c * f_b < 0:
                a, f_a = b, f_b
            else:
                f_a /= 2
            b, f_b = c, f_c
        # 5. Once |B-A| <= e, set s' <- e^(A/2)
        return math.exp(1) ** (a / 2)

    def rate(self, rating, series):
        # Step 2. For each player, convert the rating and RD's onto the
        #         Glicko-2 scale.
        rating = self.scale_down(rating)
        # Step 3. Compute the quantity v. This is the estimated variance of the
        #         team's/player's rating based only on game outcomes.
        # Step 4. Compute the quantity difference, the estimated improvement in
        #         rating by comparing the pre-period rating to the performance
        #         rating based only on game outcomes.
        variance_inv = 0
        difference = 0
        if not series:
            # If the team didn't play in the series, do only Step 6
            phi_star = math.sqrt(rating.phi**2 + rating.sigma**2)
            return self.scale_up(self.create_rating(rating.mu, phi_star, rating.sigma))
        for actual_score, other_rating in series:
            other_rating = self.scale_down(other_rating)
            impact = self.reduce_impact(other_rating)
            expected_score = self.expect_score(rating, other_rating, impact)
            variance_inv += impact**2 * expected_score * (1 - expected_score)
            difference += impact * (actual_score - expected_score)
        difference /= variance_inv
        variance = 1.0 / variance_inv
        # Step 5. Determine the new value, Sigma', ot the sigma. This
        #         computation requires iteration.
        sigma = self.determine_sigma(rating, difference, variance)
        # Step 6. Update the rating deviation to the new pre-rating period
        #         value, Phi*.
        phi_star = math.sqrt(rating.phi**2 + sigma**2)
        # Step 7. Update the rating and RD to the new values, Mu' and Phi'.
        phi = 1.0 / math.sqrt(1 / phi_star**2 + 1 / variance)
        mu = rating.mu + phi**2 * (difference / variance)
        # Step 8. Convert ratings and RD's back to original scale.
        return self.scale_up(self.create_rating(mu=mu, phi=phi, sigma=sigma))

    def rate_1vs1(self, rating1, rating2, drawn=False):
        new_rating1 = self.rate(rating1, [(DRAW if drawn else WIN, rating2)])
        new_rating2 = self.rate(rating2, [(DRAW if drawn else LOSS, rating1)])
        return new_rating1, new_rating2


class FootballData(object):
    def __init__(self, from_year: int = 2022):
        self.from_year = from_year
        self.cur_dir = os.getcwd()
        self.backup_file = os.path.join(self.cur_dir, "raw_backup.csv")

    def _tweak_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Define the columns that won't be melted
        id_vars = [
            "week",
            "day",
            "date",
            "time",
            "at_value",
            "boxscore",
            "season_year",
            "home_team",
            "away_team",
        ]

        # Define the columns that will be melted
        value_vars = ["winner", "loser"]

        # Melt the DataFrame
        df_melted = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="is_winner",
            value_name="team",
        )

        # Convert the 'is_winner' column to a boolean
        df_melted["is_winner"] = df_melted["is_winner"] == "winner"

        # Add a column for the opponent team
        df_melted["opponent"] = np.where(
            df_melted["team"] == df_melted["home_team"],
            df_melted["away_team"],
            df_melted["home_team"],
        )

        # Add a column for each week/team combination of the team they played
        df_melted["week_team_opponent"] = (
            df_melted["season_year"].astype(str)
            + "-"
            + df_melted["week"].astype(str)
            + "-"
            + df_melted["team"]
            + "-"
            + df_melted["opponent"]
        )

        return df_melted

    def fetch_data(self, from_year: int = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.backup_file)
            print("Reading backup file..")
            return self._tweak_df(df)
        except:
            print("Pulling data from web.")

        url: str = "https://www.pro-football-reference.com/years/{year}/games.htm"

        from_year = self.from_year if from_year is None else from_year

        results: list = []
        years_to_pull: int = datetime.datetime.now().year - self.from_year

        for i in range(years_to_pull):
            season_year = self.from_year + i
            df: pd.DataFrame = pd.read_html(url.format(year=season_year))[0]
            df["season_year"] = season_year
            results.append(df)

        number_of_seasons_returned = len(results)

        if number_of_seasons_returned == 1:
            results_to_process = results[0]
        elif number_of_seasons_returned > 1:
            results_to_process = pd.concat(results)
        else:
            raise Exception("No Season data returned")

        # Process the data
        df = self.process_data(results_to_process)

        return self._tweak_df(df)

    def process_data(self, df) -> pd.DataFrame:
        df = df[df["Day"] != "Day"]
        df = df[df["Date"] != "Playoffs"]

        df = df.rename(
            columns={
                "Week": "week",
                "Day": "day",
                "Date": "date",
                "Time": "time",
                "Winner/tie": "winner",
                "Unnamed: 5": "at_value",
                "Loser/tie": "loser",
                "Unnamed: 7": "boxscore",
                "Pts": "pts_winner",
                "Pts.1": "pts_loser",
                "YdsW": "yards_winner",
                "TOW": "turnovers_winner",
                "YdsL": "yards_loser",
                "TOL": "turnovers_loser",
            }
        )

        df = df.assign(
            at_value=lambda x: x.at_value.replace(np.nan, ""),
            home_team=lambda x: np.where(
                x.at_value.str.contains("@"), x.loser, x.winner
            ),
            away_team=lambda x: np.where(
                x.at_value.str.contains("@"), x.winner, x.loser
            ),
            date=lambda x: pd.to_datetime(x.date, format="%Y-%m-%d"),
            year=lambda x: x.date.dt.year,
            pts_winner=lambda x: x.pts_winner.astype(int),
            pts_loser=lambda x: x.pts_loser.astype(int),
            yards_winner=lambda x: x.yards_winner.astype(int),
            turnovers_winner=lambda x: x.turnovers_winner.astype(int),
            yards_loser=lambda x: x.yards_loser.astype(int),
            turnovers_loser=lambda x: x.turnovers_loser.astype(int),
        )

        df.to_csv(self.backup_file, index=False)

        return df


data = FootballData(2000)

games = data.fetch_data()

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


def plot_ratings(df, team_name):
    # Filter the dataframe for the specific team
    team_df = df[df["team"] == team_name]
    # Filter the dataframe for the specific year
    team_df = team_df[team_df["season_year"] > 2017]

    # Sort the dataframe by date
    team_df = team_df.sort_values(by=["season_year", "date"])

    # Create a figure and axis with increased size
    fig, ax = plt.subplots(figsize=(19, 8))  # Adjust the numbers as needed

    # Plot Glicko2 rating with each year being a different color
    sns.barplot(
        x=team_df["week"],
        y=team_df["glicko2_rating"],
        hue=team_df["season_year"],
        palette="tab10",
        ax=ax,
    )

    # Set labels
    ax.set_xlabel("Week")
    ax.set_ylabel("Glicko2 Rating")
    plt.title(f"Glicko2 Rating progression for team {team_name}")

    # Show the plot
    plt.show()


plot_ratings(df, "Minnesota Vikings")
