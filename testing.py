import pandas as pd
import datetime
import numpy as np
import math


def fetch_data(from_year: int = 2000) -> pd.DataFrame:

    url: str = "https://www.pro-football-reference.com/years/{year}/games.htm"

    results: list = []
    years_to_pull: int = datetime.datetime.now().year - from_year
    for i in range(years_to_pull):
        df: pd.DataFrame = pd.read_html(url.format(year=from_year + i))[0]
        results.append(df)

    return pd.concat(results)


# games_data: pd.DataFrame = fetch_data()


def process_data(df: pd.DataFrame) -> pd.DataFrame:

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
    )

    return df


class EloRanking:
    def __init__(self, initial_rating=1500, k_factor=32):
        self.rating = initial_rating
        self.k_factor = k_factor

    def update_rating(self, opponent_rating, outcome):
        expected_score = self._calculate_expected_score(opponent_rating)
        self.rating += self.k_factor * (outcome - expected_score)

    def _calculate_expected_score(self, opponent_rating):
        return 1 / (1 + math.pow(10, (opponent_rating - self.rating) / 400))


class GlickoRating:
    def __init__(self, initial_rating=1500, initial_rd=350, volatility=0.06):
        self.rating = initial_rating
        self.rd = initial_rd
        self.volatility = volatility

    def update_rating(self, opponent_rating, outcome):
        g = self._calculate_g(opponent_rating)
        e = self._calculate_expected_score(opponent_rating, g)
        v = self._calculate_v(g, e)
        delta = self._calculate_delta(outcome, e, v)
        self.rating += delta

        phi_star = self._calculate_phi_star(v)
        self.rd = math.sqrt(phi_star**2 + self.volatility**2)

    def _calculate_g(self, opponent_rating):
        return 1 / math.sqrt(
            1 + (3 * (self.volatility**2) * (opponent_rating**2)) / (math.pi**2)
        )

    def _calculate_expected_score(self, opponent_rating, g):
        return 1 / (1 + math.exp(-g * (self.rating - opponent_rating)))

    def _calculate_v(self, g, e):
        return 1 / (g**2 * e * (1 - e))

    def _calculate_delta(self, outcome, e, v):
        return v * g(self.rating) * (outcome - e)

    def _calculate_phi_star(self, v):
        return 1 / math.sqrt(1 / (self.rd**2) + 1 / v)


def g(x):
    return 1 / math.sqrt(1 + 3 * (x**2) / (math.pi**2))


class Glicko2Rating:
    def __init__(self, initial_rating=1500, initial_rd=350, volatility=0.06):
        self.rating = initial_rating
        self.rd = initial_rd
        self.volatility = volatility

    def update_rating(self, opponent_ratings, outcomes):
        v = self._calculate_v(opponent_ratings)
        delta = self._calculate_delta(opponent_ratings, outcomes, v)
        self.rating += delta

        phi_star = self._calculate_phi_star(v)
        self.rd = math.sqrt(self.rd**2 + phi_star**2)

    def _calculate_v(self, opponent_ratings):
        total = 0
        for opponent_rating in opponent_ratings:
            g = self._calculate_g(opponent_rating)
            e = self._calculate_expected_score(self.rating, opponent_rating, g)
            total += g**2 * e * (1 - e)
        return 1 / total

    def _calculate_delta(self, opponent_ratings, outcomes, v):
        total = 0
        for i, opponent_rating in enumerate(opponent_ratings):
            g = self._calculate_g(opponent_rating)
            e = self._calculate_expected_score(self.rating, opponent_rating, g)
            total += g * (outcomes[i] - e)
        return v * total

    def _calculate_phi_star(self, v):
        return 1 / math.sqrt(1 / (self.rd**2) + 1 / v)

    def _calculate_g(self, opponent_rating):
        return 1 / math.sqrt(
            1 + 3 * (self.volatility**2) * (opponent_rating**2) / (math.pi**2)
        )

    def _calculate_expected_score(self, player_rating, opponent_rating, g):
        return 1 / (1 + math.exp(-g * (player_rating - opponent_rating)))


# Example usage
team1_elo = EloRanking()
team2_elo = EloRanking()

team1_elo.update_rating(team2_elo.rating, 1)  # team1 wins
team2_elo.update_rating(team1_elo.rating, 0)  # team2 loses

team1_glicko = GlickoRating()
team2_glicko = GlickoRating()

team1_glicko.update_rating(team2_glicko.rating, 1)  # team1 wins
team2_glicko.update_rating(team1_glicko.rating, 0)  # team2 loses

team1_glicko2 = Glicko2Rating()
team2_glicko2 = Glicko2Rating()

team1_glicko2.update_rating([team2_glicko2.rating], [1])  # team1 wins
team2_glicko2.update_rating([team1_glicko2.rating], [0])  # team2 loses

print(f"Team 1 Elo: {team1_elo.rating}")
print(f"Team 2 Elo: {team2_elo.rating}")

print(f"Team 1 Glicko: {team1_glicko.rating}")
print(f"Team 2 Glicko: {team2_glicko.rating}")

print(f"Team 1 Glicko2: {team1_glicko2.rating}")
print(f"Team 2 Glicko2: {team2_glicko2.rating}")
