import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_impact(df):
    top_pos = df[df['Impact_log'] > 0].sort_values('Impact_log', ascending=False).head(5)
    top_neg = df[df['Impact_log'] < 0].sort_values('Impact_log').head(5)

    plot_df = pd.concat([top_pos, top_neg]).sort_values('Impact_percent')

    colors = ['green' if x > 0 else 'red' for x in plot_df['Impact_percent']]

    fig = plt.figure(figsize=(8,5))
    plt.barh(plot_df['Feature'], plot_df['Impact_percent'], color=colors)
    plt.xlabel("Impact on Price (%)")
    plt.title("Top Positive & Negative Factors")
    plt.gca().invert_yaxis()

    return fig