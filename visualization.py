import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def visualize_results(results_df, market_df, call_date):
    fig, axs = plt.subplots(3, 1, figsize=(13, 12))

    sentiment_keys = ['pos_minus_neg', 'compound', 'lm_score']
    sentiment_labels = ['FinBERT (Pos-Neg)', 'VADER Compound', 'LM Score']
    scores = [results_df[k][0] for k in sentiment_keys]
    axs[0].barh(sentiment_labels, scores, color=['royalblue', 'seagreen', 'darkorange'])
    axs[0].set_title('Earnings Call Sentiment Scores')
    axs[0].set_xlim(-1, 1)
    for i, v in enumerate(scores):
        axs[0].text(v, i, f"{v:.2f}", color='black', va='center', ha='left' if v >= 0 else 'right', fontsize=12)

    axs[1].plot(market_df.index, market_df['Close'], marker='o', linestyle='-', color='navy', label='Close Price')
    axs[1].axvline(pd.to_datetime(call_date), color='red', linestyle='--', lw=2, label='Earnings Call')
    axs[1].set_title('AAPL Closing Price Around Earnings Call')
    axs[1].set_ylabel('Close Price (USD)')
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[1].legend()
    axs[1].tick_params(axis='x', rotation=45)

    axs[2].plot(market_df.index, market_df['Volume'], marker='o', linestyle='-', color='purple', label='Volume')
    axs[2].axvline(pd.to_datetime(call_date), color='red', linestyle='--', lw=2, label='Earnings Call')
    axs[2].set_title('AAPL Trading Volume Around Earnings Call')
    axs[2].set_ylabel('Volume')
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[2].legend()
    axs[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show(block=True)
    print("Plots displayed. Close the plot window to finish the script.")

def print_summary(results_df):
    print("="*50)
    print("SENTIMENT & MARKET REACTION REPORT")
    print("="*50)
    print(f"VADER compound score: {results_df['compound'][0]:.3f}")
    print(f"FinBERT pos-neg score: {results_df['pos_minus_neg'][0]:.3f}")
    print(f"LM Wordlist score: {results_df['lm_score'][0]:.3f}")
    ndc = results_df['next_day_close'][0]
    ndv = results_df['next_day_volume'][0]
    if ndc is not None:
        print(f"\nAAPL Close after call: {ndc:.2f}")
    else:
        print("\nAAPL Close after call: N/A")
    if ndv is not None:
        print(f"AAPL Volume after call: {int(ndv):,d}")
    else:
        print("AAPL Volume after call: N/A")
    print("="*50)

