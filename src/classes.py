import pandas as pd
import datetime
import numpy as np
import math
from tqdm import tqdm
import logging
import functools
import time
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#: The actual score for win
WIN = 1.
#: The actual score for draw
DRAW = 0.5
#: The actual score for loss
LOSS = 0.

K_FACTOR = 32


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
    def __init__(self, mu=MU, phi=PHI, sigma=SIGMA):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    def __repr__(self):
        c = type(self)
        args = (c.__module__, c.__name__, self.mu, self.phi, self.sigma)
        return '%s.%s(mu=%.3f, phi=%.3f, sigma=%.3f)' % args

    def rate(self):
        raise NotImplemented

    def rate_1vs1(self):
        raise NotImplemented


class EloRating(object):
    def __init__(self, mu=MU, k_factor=K_FACTOR):
        self.k_factor = k_factor
        self.type = 'Elo'
        self.mu = mu

    def create_rating(self, mu=MU):
        return Rating(MU)

    def rate(self, rating1, rating2, outcome):
        expected_score = self._calculate_expected_score(rating1, rating2)
        delta = self.k_factor * (outcome - expected_score)
        return Rating(mu=rating1.mu + delta)

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
        self.type = 'Glicko2'

    def create_rating(self, mu=None, phi=None, sigma=None):
        if mu is None:
            mu = self.mu
        if phi is None:
            phi = self.phi
        if sigma is None:
            sigma = self.sigma
        return Rating(mu, phi, sigma)

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
        return 1. / math.sqrt(1 + (3 * rating.phi ** 2) / (math.pi ** 2))

    def expect_score(self, rating, other_rating, impact):
        return 1. / (1 + math.exp(-impact * (rating.mu - other_rating.mu)))

    def determine_sigma(self, rating, difference, variance):
        """Determines new sigma."""
        phi = rating.phi
        difference_squared = difference ** 2
        # 1. Let a = ln(s^2), and define f(x)
        alpha = math.log(rating.sigma ** 2)

        def f(x):
            """This function is twice the conditional log-posterior density of
            phi, and is the optimality criterion.
            """
            tmp = phi ** 2 + variance + math.exp(x)
            a = math.exp(x) * (difference_squared - tmp) / (2 * tmp ** 2)
            b = (x - alpha) / (self.tau ** 2)
            return a - b

        # 2. Set the initial values of the iterative algorithm.
        a = alpha
        if difference_squared > phi ** 2 + variance:
            b = math.log(difference_squared - phi ** 2 - variance)
        else:
            k = 1
            while f(alpha - k * math.sqrt(self.tau ** 2)) < 0:
                k += 1
            b = alpha - k * math.sqrt(self.tau ** 2)
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
            phi_star = math.sqrt(rating.phi ** 2 + rating.sigma ** 2)
            return self.scale_up(self.create_rating(rating.mu, phi_star, rating.sigma))
        for actual_score, other_rating in series:
            other_rating = self.scale_down(other_rating)
            impact = self.reduce_impact(other_rating)
            expected_score = self.expect_score(rating, other_rating, impact)
            variance_inv += impact ** 2 * expected_score * (1 - expected_score)
            difference += impact * (actual_score - expected_score)
        difference /= variance_inv
        variance = 1. / variance_inv
        # Step 5. Determine the new value, Sigma', ot the sigma. This
        #         computation requires iteration.
        sigma = self.determine_sigma(rating, difference, variance)
        # Step 6. Update the rating deviation to the new pre-rating period
        #         value, Phi*.
        phi_star = math.sqrt(rating.phi ** 2 + sigma ** 2)
        # Step 7. Update the rating and RD to the new values, Mu' and Phi'.
        phi = 1. / math.sqrt(1 / phi_star ** 2 + 1 / variance)
        mu = rating.mu + phi ** 2 * (difference / variance)
        # Step 8. Convert ratings and RD's back to original scale.
        return self.scale_up(self.create_rating(mu, phi, sigma))

    def rate_1vs1(self, rating1, rating2, drawn=False):
        return (self.rate(rating1, [(DRAW if drawn else WIN, rating2)]),
                self.rate(rating2, [(DRAW if drawn else LOSS, rating1)]))


    def quality_1vs1(self, rating1, rating2):
        expected_score1 = self.expect_score(rating1, rating2, self.reduce_impact(rating1))
        expected_score2 = self.expect_score(rating2, rating1, self.reduce_impact(rating2))
        expected_score = (expected_score1 + expected_score2) / 2
        return 2 * (0.5 - abs(0.5 - expected_score))


class FootballData(object):

    def __init__(self, from_year: int = 2022):
        self.from_year = from_year

    @functools.lru_cache(maxsize=4)
    def fetch_data(self, from_year:int=None) -> pd.DataFrame:
        url: str = "https://www.pro-football-reference.com/years/{year}/games.htm"

        from_year = self.from_year if from_year is None else from_year

        results: list = []
        years_to_pull: int = datetime.datetime.now().year - self.from_year

        for i in range(years_to_pull):

            df: pd.DataFrame = pd.read_html(url.format(year=self.from_year + i))[0]
            results.append(df)

        number_of_seasons_returned = len(results)

        if number_of_seasons_returned == 1:
            results_to_process = results[0]
        elif number_of_seasons_returned > 1:
            results_to_process = pd.concat(results)
        else:
            raise Exception("No Season data returned")
        
        return_val = self.process_data(results_to_process)

        return return_val



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
            home_team=lambda x: np.where(x.at_value.str.contains("@"), x.loser, x.winner),
            away_team=lambda x: np.where(x.at_value.str.contains("@"), x.winner, x.loser),
            date=lambda x: pd.to_datetime(x.date, format="%Y-%m-%d"),
            year=lambda x: x.date.dt.year,
            pts_winner=lambda x: x.pts_winner.astype(int),
            pts_loser=lambda x: x.pts_loser.astype(int),
            yards_winner=lambda x: x.yards_winner.astype(int),
            turnovers_winner=lambda x: x.turnovers_winner.astype(int),
            yards_loser=lambda x: x.yards_loser.astype(int),
            turnovers_loser=lambda x: x.turnovers_loser.astype(int)
        )

        df = df[[x for x in df.columns if x not in ['home_team','away_team']]]

        df = df.rename(columns={
            'winner': 'team_winner',
            'loser': 'team_loser',
            'pts_winner': 'pts_team_winner',
            'pts_loser': 'pts_team_loser',
            'yards_winner': 'yards_team_winner',
            'yards_loser': 'yards_team_loser',
            'turnovers_winner': 'turnovers_team_winner',
            'turnovers_loser': 'turnovers_team_loser',
        })

        # Split DataFrame into two: one for home teams and one for away teams
        winner_df = df[['week', 'day', 'date', 'time', 'team_winner',  'team_loser', 'pts_team_winner', 'pts_team_loser', 'yards_team_winner', 'turnovers_team_winner', 'year']]
        loser_df = df[['week', 'day', 'date', 'time', 'team_loser', 'team_winner', 'pts_team_loser', 'pts_team_winner','yards_team_loser', 'turnovers_team_loser', 'year']]

        # Rename columns so they match
        winner_df = (winner_df
            .assign(
                draw = lambda df: df.pts_team_winner == df.pts_team_loser,
                outcome = lambda df: df.draw.apply(lambda x: 'draw' if x else 'win')
            )
            .rename(
                columns={
                    'team_winner': 'team',
                    'team_loser': 'opponent',
                    'pts_team_winner': 'pts',
                    'yards_team_winner': 'yards',
                    'turnovers_team_winner': 'turnovers'
                }
            )
        )
        loser_df = (loser_df
            .assign(
                draw = lambda df: df.pts_team_winner == df.pts_team_loser,
                outcome = lambda df: df.draw.apply(lambda x: 'draw' if x else 'loss')
            )
            .rename(
            columns={
                'team_loser': 'team',
                'team_winner': 'opponent',
                'pts_team_loser': 'pts',
                'yards_team_loser': 'yards',
                'turnovers_team_loser': 'turnovers'
            })
        )
        # Combine winner_df and loser_df
        final_df = pd.concat([winner_df, loser_df]).drop(['pts_team_loser','pts_team_winner','draw'], axis='columns')

        # Sort by date to get the games in order
        final_df = final_df.sort_values('date')

        # Reset index
        final_df = final_df.reset_index(drop=True)

        return final_df



data = FootballData(2000)

games = data.fetch_data()



glicko2 = Glicko2()
elo = EloRating()

# Create dictionaries to store the ratings for each team
glicko2_ratings = {}
elo_ratings = {}

df = games.copy()

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    team = row['team']
    opponent = row['opponent']
    outcome = row['outcome']

    # If the team or the opponent is not already in the dictionary, create a new rating
    for team_name, rating_dict in [(team, glicko2_ratings), (opponent, glicko2_ratings), (team, elo_ratings), (opponent, elo_ratings)]:
        if team_name not in rating_dict:
            rating_dict[team_name] = glicko2.create_rating() if rating_dict is glicko2_ratings else elo.create_rating()

    drawn = False
    # Update the ratings if the team won
    if outcome == 'win':
        glicko2_ratings[team], glicko2_ratings[opponent] = glicko2.rate_1vs1(glicko2_ratings[team], glicko2_ratings[opponent], drawn=drawn)
        elo_ratings[team], elo_ratings[opponent] = elo.rate_1vs1(elo_ratings[team], elo_ratings[opponent], drawn=drawn)
    elif outcome == 'loss':
        glicko2_ratings[opponent], glicko2_ratings[team] = glicko2.rate_1vs1(glicko2_ratings[opponent], glicko2_ratings[team], drawn=drawn)
        elo_ratings[team], elo_ratings[opponent] = elo.rate_1vs1(elo_ratings[opponent], elo_ratings[team], drawn=drawn)
    # Update the ratings if the game is drawn
    elif outcome == 'draw': # assuming that 'draw' is the word used in your dataframe for draws
        drawn = True
        glicko2_ratings[team], glicko2_ratings[opponent] = glicko2.rate_1vs1(glicko2_ratings[team], glicko2_ratings[opponent], drawn=drawn)
        elo_ratings[team], elo_ratings[opponent] = elo.rate_1vs1(elo_ratings[team], elo_ratings[opponent], drawn=drawn)

# Add the ratings as new columns in the DataFrame
df['glicko2_rating'] = df['team'].map(lambda team: glicko2_ratings[team].mu)
df['elo_rating'] = df['team'].map(lambda team: elo_ratings[team].mu)


df

