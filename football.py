#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 2025

@author: tunahan

martj42/international-football-results-from-1872-to-2017

Description: Compute and visualize international soccer teams’ Elo ratings
and various match statistics.
The code was refactored and modularized with the help of AI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ------------------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------------------
DATA_DIR = "./data"
FORMER_NAMES_FILE = f"{DATA_DIR}/former_names.csv"
GOALSCORERS_FILE   = f"{DATA_DIR}/goalscorers.csv"
MATCHES_FILE       = f"{DATA_DIR}/results.csv"
SHOOTOUTS_FILE     = f"{DATA_DIR}/shootouts.csv"

# Tournament categories for K-factor
WORLD_CUP_AND_OLYMPICS = [
    "FIFA World Cup", "Olympic Games"
]
CONTINENTAL_TOURNAMENTS = [
    "Copa América", "UEFA Euro", "AFC Asian Cup", "African Cup of Nations",
    "Oceania Nations Cup", "CONCACAF Championship", "Gold Cup",
    "Confederations Cup", "EAFF Championship", "AFF Championship",
    "SAFF Cup", "WAFF Championship", "CAFA Nations Cup",
    "CONIFA World Football Cup", "CONMEBOL–UEFA Cup of Champions"
]
QUALIFICATION_CATEGORIES = [
    "qualification", "FIFA World Cup qualification", "UEFA Euro qualification",
    "AFC Asian Cup qualification", "African Cup of Nations qualification",
    "CONCACAF Championship qualification", "Gold Cup qualification",
    "AFF Championship qualification", "Copa América qualification",
    "Oceania Nations Cup qualification", "CONCACAF Nations League",
    "CONCACAF Nations League qualification", "EAFF Championship qualification",
    "ASEAN Championship qualification"
]

# ------------------------------------------------------------------------------
#  UTILITY FUNCTIONS
# ------------------------------------------------------------------------------
def load_and_normalize_data():
    """Load CSVs and normalize team names using the former_names mapping."""
    former_df    = pd.read_csv(FORMER_NAMES_FILE)
    name_mapping = dict(zip(former_df.former, former_df.current))

    goals_df   = pd.read_csv(GOALSCORERS_FILE)
    matches_df = pd.read_csv(MATCHES_FILE)
    shoot_df   = pd.read_csv(SHOOTOUTS_FILE)

    for df in (goals_df, matches_df, shoot_df):
        df.replace(name_mapping, inplace=True)

    return goals_df, matches_df, shoot_df

def expected_score(elo_a, elo_b, is_home=False):
    """Compute expected score given two Elo ratings (home advantage = +100)."""
    diff = elo_a - elo_b + (100 if is_home else 0)
    return 1 / (1 + 10 ** (-diff / 400))

def goal_diff_multiplier(goal_diff):
    """Return G multiplier based on goal difference."""
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8

def get_k_factor(tournament_name):
    """Assign K-factor based on tournament importance."""
    name = tournament_name.lower()
    if any(t.lower() == name for t in WORLD_CUP_AND_OLYMPICS):
        return 60
    if any(t.lower() == name for t in CONTINENTAL_TOURNAMENTS):
        return 40
    if "qualification" in name or any(t.lower() == name for t in QUALIFICATION_CATEGORIES):
        return 30
    if name == "friendly":
        return 15
    return 25

def update_elo(current_elo, actual_score, expected, k, multiplier=1.0):
    """Update Elo rating."""
    return current_elo + k * multiplier * (actual_score - expected)

# ------------------------------------------------------------------------------
#  ELO COMPUTATION
# ------------------------------------------------------------------------------
def compute_elo_history(matches_df):
    """
    Iterate through sorted matches and produce a history of Elo updates.
    Returns final elo_dict and a list of per‐match history records.
    """
    team_counts = Counter(matches_df.home_team).copy()
    team_counts.update(matches_df.away_team)

    # Initialize all teams at 1500
    elo_dict = {team: 1500 for team in pd.concat([matches_df.home_team,
                                                   matches_df.away_team]).unique()}

    history = []
    for _, match in matches_df.sort_values("date").iterrows():
        home, away = match.home_team, match.away_team
        home_elo, away_elo = elo_dict[home], elo_dict[away]

        # Determine match result
        h_score, a_score = match.home_score, match.away_score
        if h_score > a_score:
            result_home, result_away = 1.0, 0.0
        elif h_score < a_score:
            result_home, result_away = 0.0, 1.0
        else:
            result_home = result_away = 0.5

        # Compute parameters
        k = get_k_factor(match.tournament)
        g = goal_diff_multiplier(abs(h_score - a_score))
        exp_home = expected_score(home_elo, away_elo, is_home=True)
        exp_away = 1 - exp_home

        # Update Elo ratings
        new_home_elo = update_elo(home_elo, result_home, exp_home, k, g)
        new_away_elo = update_elo(away_elo, result_away, exp_away, k, g)
        elo_dict[home] = new_home_elo
        elo_dict[away] = new_away_elo

        # Record history
        history.append({
            "date":       match.date,
            "home_team":  home,
            "away_team":  away,
            "home_elo":   new_home_elo,
            "away_elo":   new_away_elo,
            "home_score": h_score,
            "away_score": a_score,
            "k_factor":   k,
            "g_factor":   g
        })

    # Filter out teams with fewer than 300 matches
    eligible = {t: e for t, e in elo_dict.items() if team_counts[t] > 300}
    return eligible, history, team_counts

# ------------------------------------------------------------------------------
#  VISUALIZATION FUNCTIONS
# ------------------------------------------------------------------------------
def plot_top_elos(elo_dict, team_counts, top_n=20):
    """Bar chart of the top N teams by Elo rating."""
    df = pd.DataFrame([
        {"team": team, "elo": round(elo), "matches": team_counts[team]}
        for team, elo in elo_dict.items()
    ])
    df = df.sort_values("elo", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="elo", y="team", palette="Reds_r", hue=df["team"])
    plt.title(f"Top {top_n} Teams by Elo Rating")
    plt.xlabel("Elo Rating")
    plt.ylabel("Team")
    plt.tight_layout()
    plt.show()

def plot_team_elo_over_time(history, team1, team2=None):
    """Line plot of one or two teams’ Elo ratings over time."""
    df = pd.DataFrame(history)
    df.date = pd.to_datetime(df.date)

    plt.figure(figsize=(12, 6))
    for team, marker in [(team1, "o"), (team2, "x")]:
        if not team:
            continue
        team_df = df[(df.home_team == team) | (df.away_team == team)].copy()
        team_df["elo"] = np.where(
            team_df.home_team == team, 
            team_df.home_elo, 
            team_df.away_elo
        )
        plt.plot(team_df.date, team_df.elo, label=team, marker=marker)
    plt.title(f"Elo Over Time: {team1}" + (f" vs {team2}" if team2 else ""))
    plt.xlabel("Date")
    plt.ylabel("Elo Rating")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_yearly_leaders(history):
    """
    Plot the highest Elo achieved by any team at the end of each year.
    """
    df = pd.DataFrame(history)
    df['date'] = pd.to_datetime(df['date'])

    # For each match record, compute the maximum Elo on that date
    df['max_elo'] = df[['home_elo', 'away_elo']].max(axis=1)

    # Find, for each year, the single record where max_elo is highest
    end_of_year = (
        df
        .groupby(df['date'].dt.year, group_keys=False)
        .apply(lambda g: g.loc[g['max_elo'].idxmax()])
    )

    # Build a summary of (year, team, elo)
    summary = pd.DataFrame({
        'year': end_of_year['date'].dt.year,
        'team': np.where(
            end_of_year['home_elo'] > end_of_year['away_elo'],
            end_of_year['home_team'],
            end_of_year['away_team']
        ),
        'elo': end_of_year['max_elo'].round().astype(int)
    })

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary, x='year', y='elo', marker='o', linewidth=3)
    for _, row in summary.iterrows():
        plt.text(row['year'], row['elo'] + 5, row['team'],
                 ha='center', fontsize=10, fontweight='bold')
    plt.title("Yearly Elo Leader")
    plt.xlabel("Year")
    plt.ylabel("Elo Rating")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_goal_distribution(goals_df, top_k=20):
    """Bar chart of the top K minutes by goal frequency."""
    freq = goals_df.minute.value_counts().head(top_k).sort_values(ascending=False)
    avg = goals_df.minute.value_counts().mean()

    plt.figure(figsize=(12, 5))
    sns.barplot(x=freq.index.astype(str), y=freq.values, palette="Blues_r", hue=freq.values)
    plt.axhline(avg, linestyle="--", label=f"Average per minute: {avg:.2f}")
    plt.title(f"Top {top_k} Minutes by Goal Count")
    plt.xlabel("Minute")
    plt.ylabel("Goals Scored")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heatmaps(matches_df, top_teams):
    """
    Two heatmaps:
      1) Clean sheets percentage for top teams
      2) Home vs. away win rates
    """
    # Clean sheet percentages
    clean_pct = {}
    for team in top_teams:
        home_cs = matches_df[(matches_df.home_team == team) & (matches_df.away_score == 0)].shape[0]
        away_cs = matches_df[(matches_df.away_team == team) & (matches_df.home_score == 0)].shape[0]
        total_matches = matches_df[(matches_df.home_team == team) | 
                                   (matches_df.away_team == team)].shape[0]
        clean_pct[team] = 100 * (home_cs + away_cs) / total_matches if total_matches else 0

    clean_df = pd.DataFrame(clean_pct, index=["Clean Sheet %"]).T

    # Home vs away win rate
    win_rates = {}
    for team in top_teams:
        home = matches_df[matches_df.home_team == team]
        away = matches_df[matches_df.away_team == team]
        win_rates[team] = [
            100 * (home.home_score > home.away_score).mean() if not home.empty else 0,
            100 * (away.away_score > away.home_score).mean() if not away.empty else 0
        ]

    wins_df = pd.DataFrame(win_rates, index=["Home Win %", "Away Win %"]).T

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(clean_df, annot=True, fmt=".1f", ax=axes[0], cbar=False)
    axes[0].set_title("Clean Sheet Percentage")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(wins_df, annot=True, fmt=".1f", ax=axes[1], cbar=False)
    axes[1].set_title("Home vs Away Win Rate")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
#  MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    # Load and normalize data
    goals_df, matches_df, shootout_df = load_and_normalize_data()

    # Compute Elo
    elo_dict, history, team_counts = compute_elo_history(matches_df)
    

    # Visualizations
    plot_top_elos(elo_dict, team_counts, top_n=20)
    plot_team_elo_over_time(history, "Turkey", "Germany")
    plot_yearly_leaders(history)
    plot_goal_distribution(goals_df, top_k=20)

    # Identify top 20 for heatmaps
    top_20 = sorted(elo_dict, key=elo_dict.get, reverse=True)[:20]
    plot_heatmaps(matches_df, top_20)

if __name__ == "__main__":
    main()
