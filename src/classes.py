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

# todo: restructure end-result to not be home vs away
#: The actual score for win
WIN = 1.
#: The actual score for draw
DRAW = 0.5
#: The actual score for loss
LOSS = 0.

K_FACTOR = 32


MU = 1500
SIGMA = 350
#: A constant which is used to standardize the logistic function to
#: `1/(1+exp(-x))` from `1/(1+10^(-r/400))`
Q = math.log(10) / 400


# Glicko related
VOLATILITY = 0.06
TAU = 1.0
EPSILON = 0.000001
DELTA = 0.0001
PHI = 350


def utctime():
    """A function like :func:`time.time` but it uses a time of UTC."""
    return time.mktime(datetime.datetime.utcnow().timetuple())


class BaseRating:
    def __init__(self, initial_rating=MU):
        self.rating = initial_rating

    def update_rating(self, opponent_rating, outcome):
        raise NotImplementedError("Subclasses must implement this method.")

    def _calculate_g(self, opponent_rating):
        return 1 / math.sqrt(1 + 3 * (opponent_rating**2) / (math.pi**2))


class EloRating(BaseRating):
    def __init__(self, initial_rating=MU, k_factor=K_FACTOR):
        super().__init__(initial_rating)
        self.k_factor = k_factor

    def update_rating(self, opponent_rating, outcome):
        expected_score = self._calculate_expected_score(opponent_rating)
        self.rating += self.k_factor * (outcome - expected_score)

    def _calculate_expected_score(self, opponent_rating):
        return 1 / (1 + math.pow(10, (opponent_rating - self.rating) / 400))


class Rating:
    def __init__(self, mu: float = MU, phi: float = PHI, sigma: float = SIGMA):
        # TODO add a name
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    def __repr__(self):
        c = type(self)
        args = (c.__module__, c.__name__, self.mu, self.phi, self.sigma)
        return '%s.%s(mu=%.3f, phi=%.3f, sigma=%.3f)' % args


class Glicko2:
    def __init__(self, mu: float = MU, phi: float = PHI, sigma: float = SIGMA, tau: float = TAU, epsilon: float = EPSILON):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.tau = tau
        self.epsilon = epsilon

    def create_rating(self, mu: Optional[float] = None, phi: Optional[float] = None, sigma: Optional[float] = None) -> Rating:
        if mu is None:
            mu = self.mu
        if phi is None:
            phi = self.phi
        if sigma is None:
            sigma = self.sigma
        return Rating(mu, phi, sigma)

    def scale_down(self, rating: Rating, ratio: float = 173.7178) -> Rating:
        mu = (rating.mu - self.mu) / ratio
        phi = rating.phi / ratio
        return self.create_rating(mu, phi, rating.sigma)

    def scale_up(self, rating: Rating, ratio: float = 173.7178) -> Rating:
        mu = rating.mu * ratio + self.mu
        phi = rating.phi * ratio
        return self.create_rating(mu, phi, rating.sigma)

    def reduce_impact(self, rating: Rating) -> float:
        return 1. / math.sqrt(1 + (3 * rating.phi ** 2) / (math.pi ** 2))

    def expect_score(self, rating: Rating, other_rating: Rating, impact: float) -> float:
        return 1. / (1 + math.exp(-impact * (rating.mu - other_rating.mu)))

    def determine_sigma(self, rating: Rating, difference: float, variance: float) -> float:
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

    def rate(self, rating: Rating, series: List[Tuple[float, Rating]]) -> Rating:
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

    def rate_1vs1(self, winner: Rating, loser: Rating, drawn: bool = False) -> Tuple[Rating, Rating]:
        return (self.rate(winner, [(DRAW if drawn else WIN, loser)]),
                self.rate(loser, [(DRAW if drawn else LOSS, winner)]))

    def quality_1vs1(self, rating1: Rating, rating2: Rating) -> float:
        expected_score1 = self.expect_score(rating1, rating2, self.reduce_impact(rating1))
        expected_score2 = self.expect_score(rating2, rating1, self.reduce_impact(rating2))
        expected_score = (expected_score1 + expected_score2) / 2
        return 2 * (0.5 - abs(0.5 - expected_score))


class FootballData(object):

    def __init__(self, from_year: int = 2000):
        self.from_year = from_year
        self.games = self.process_data()

    def fetch_data(self) -> pd.DataFrame:
        url: str = "https://www.pro-football-reference.com/years/{year}/games.htm"

        results: list = []
        years_to_pull: int = datetime.datetime.now().year - self.from_year
        for i in range(years_to_pull):
            df: pd.DataFrame = pd.read_html(
                url.format(year=self.from_year + i))[0]
            results.append(df)

        return pd.concat(results)

    @functools.lru_cache(maxsize=1)
    def process_data(self) -> pd.DataFrame:
        df = self.fetch_data()
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
                x.at_value.str.contains("@"), x.loser, x.winner),
            away_team=lambda x: np.where(
                x.at_value.str.contains("@"), x.winner, x.loser),
            date=lambda x: pd.to_datetime(x.date, format="%Y-%m-%d"),
            year=lambda x: x.date.dt.year,
        )

        return df


data = FootballData()

elo_ratings = {}  # dict to hold EloRating objects for each team
glicko_ratings = {}  # dict to hold GlickoRating objects for each team
glicko2_ratings = {}  # dict to hold Glicko2Rating objects for each team

# For storing game results and ratings
game_results = []
glicko2 = Glicko2()

for index, row in tqdm(data.games.iterrows(), total=data.games.shape[0]):
    home_team = row['home_team']
    away_team = row['away_team']
    winner = row['winner']
    loser = row['loser']

    if winner not in elo_ratings:

        elo_ratings[winner] = EloRating()
        glicko2_ratings[winner] = glicko2.create_rating()

    if loser not in elo_ratings:
        elo_ratings[loser] = EloRating()
        glicko2_ratings[loser] = glicko2.create_rating()

    if winner == home_team:
        outcome_home = WIN
        outcome_away = LOSS
        drawn = False
    elif winner == away_team:
        outcome_home = LOSS
        outcome_away = WIN
        drawn = False
    else:  # draw
        outcome_home = 0.5
        drawn = True

    elo_ratings[home_team].update_rating(elo_ratings[away_team].rating, outcome_home)
    elo_ratings[away_team].update_rating(elo_ratings[home_team].rating, outcome_away)

    glicko_ratings[winner], glicko_ratings[loser] = glicko2.rate_1vs1(glicko2_ratings[winner], glicko2_ratings[loser], drawn=drawn)
    glicko_ratings[home_team] == glicko_ratings[winner] if home_team == winner else glicko_ratings[loser]
    glicko_ratings[away_team] == glicko_ratings[winner] if away_team == winner else glicko_ratings[loser]
    # glicko2_ratings[home_team] = glicko2_ratings[home_team].rate_1vs1(glicko2_ratings[away_team], outcome_home == 0.5)[0]
    # glicko2_ratings[away_team] = glicko2_ratings[away_team].rate_1vs1(glicko2_ratings[home_team], outcome_home == 0.5)[1]

    game_results.append({
        'home_team': home_team,
        'away_team': away_team,
        'outcome_home': outcome_home,
        'outcome_away': outcome_away,
        'home_elo': elo_ratings[home_team].rating,
        'away_elo': elo_ratings[away_team].rating,
        'home_glicko': glicko_ratings[home_team].mu,
        'away_glicko': glicko_ratings[away_team].mu,
        # 'home_glicko2': glicko2_ratings[home_team].mu,
        # 'away_glicko2': glicko2_ratings[away_team].mu
    })

results_df = pd.DataFrame(game_results)
