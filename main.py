from preprocessing import load_transcript, clean_text, fetch_market_data, align_market_data_with_call
from sentiment import run_vader, run_finbert, run_loughran_mcdonald
from integration import integrate_data
from visualization import visualize_results, print_summary

if __name__ == "__main__":
    transcript_path = "AAPL.txt"
    ticker = "AAPL"
    call_date = "2025-04-24"
    raw_text = load_transcript(transcript_path)
    clean = clean_text(raw_text)
    market_df = fetch_market_data(ticker, "2025-04-20", "2025-04-30")
    market_reaction = align_market_data_with_call(market_df, call_date)
    vader_scores = run_vader(raw_text)
    finbert_scores = run_finbert(raw_text)
    lm_scores = run_loughran_mcdonald(clean)
    results_df = integrate_data(vader_scores, finbert_scores, lm_scores, market_reaction)
    print_summary(results_df)
    visualize_results(results_df, market_df, call_date)

