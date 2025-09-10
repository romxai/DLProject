# d:/Users/Fiona/Desktop/projects/DLProject/main.py
import asyncio
import os
import pandas as pd
from datetime import datetime
from src.scraping.news_scraper import scrape_news
from src.scraping.twitter_scraper import scrape_twitter
from src.processing.sentiment_analyzer import SentimentAnalyzer
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)

async def main():
    """
    Main function to run the scraping and processing pipeline.
    """
    # --- Configuration ---
    # Stock symbols to analyze
    STOCK_SYMBOLS = ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN"]
    
    # Which stock to process in this run
    CURRENT_STOCK = "TSLA"  # Can be changed to any symbol in STOCK_SYMBOLS
    
    # News sources configuration
    NEWS_SOURCES = {
        "TSLA": [
            "https://www.cnbc.com/quotes/TSLA",
            "https://finance.yahoo.com/quote/TSLA"
        ],
        "AAPL": [
            "https://www.cnbc.com/quotes/AAPL",
            "https://finance.yahoo.com/quote/AAPL"
        ],
        "MSFT": [
            "https://www.cnbc.com/quotes/MSFT",
            "https://finance.yahoo.com/quote/MSFT"
        ],
        "GOOGL": [
            "https://www.cnbc.com/quotes/GOOGL",
            "https://finance.yahoo.com/quote/GOOGL"
        ],
        "AMZN": [
            "https://www.cnbc.com/quotes/AMZN",
            "https://finance.yahoo.com/quote/AMZN"
        ]
    }
    
    # Twitter configuration
    TWITTER_QUERIES = {
        "TSLA": ["$TSLA", "Tesla", "Elon Musk Tesla"],
        "AAPL": ["$AAPL", "Apple", "Tim Cook Apple"],
        "MSFT": ["$MSFT", "Microsoft", "Satya Nadella Microsoft"],
        "GOOGL": ["$GOOGL", "Google", "Alphabet"],
        "AMZN": ["$AMZN", "Amazon", "Andy Jassy Amazon"]
    }
    
    # Chrome user profile settings
    USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "chrome-data", "twitter-profile")
    HEADLESS = False  # Set to True once profile is set up
    
    # Scraping settings
    NUM_TWEETS = 30
    MAX_NEWS_ARTICLES = 50
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # --- Scraping Layer ---
    logging.info(f"Starting Scraping Layer for {CURRENT_STOCK}")
    
    # Only do Twitter scraping
    TWITTER_QUERIES = {
        "TSLA": ["$TSLA", "Tesla", "Elon Musk Tesla"],
    }
    NUM_TWEETS = 30
    USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "chrome-data", "twitter-profile")
    HEADLESS = False
    twitter_tasks = []
    for query in TWITTER_QUERIES.get(CURRENT_STOCK, []):
        logging.debug(f"Adding Twitter scraping task for query: {query}")
        twitter_tasks.append(
            scrape_twitter(
                query, 
                num_tweets=NUM_TWEETS, 
                user_data_dir=USER_DATA_DIR,
                headless=HEADLESS
            )
        )
    twitter_results = await asyncio.gather(*twitter_tasks)
    logging.info(f"Completed Twitter scraping. Total queries: {len(twitter_results)}")
    all_tweets_df = pd.concat(twitter_results, ignore_index=True) if twitter_results else pd.DataFrame()
    logging.info(f"Total tweets scraped: {len(all_tweets_df)}")
    if not all_tweets_df.empty:
        tweets_file = os.path.join(output_dir, f"{CURRENT_STOCK}_tweets_raw_{run_timestamp}.csv")
        all_tweets_df.to_csv(tweets_file, index=False)
        logging.info(f"Saved raw tweets data to {tweets_file}")

    # --- Filtering Layer ---
    def is_spam_or_irrelevant(tweet):
        text = tweet.get('text', '')
        # Remove tweets with excessive hashtags, links, or promotional phrases
        if text.count('#') > 4 or text.count('http') > 1:
            return True
        if any(kw in text.lower() for kw in ['earn $', 'follow him now', 'finance professor', 'stock picks are gold', 'buy now', 'free trial']):
            return True
        if len(text) < 20 or len(text) > 300:
            return True
        # Remove tweets that are mostly mentions
        if text.strip().startswith('@') and text.count('@') > 2:
            return True
        return False
    
    # Remove duplicates and spam
    all_tweets_df.drop_duplicates(subset=['text'], inplace=True)
    filtered_tweets_df = all_tweets_df[~all_tweets_df.apply(is_spam_or_irrelevant, axis=1)].copy()
    filtered_tweets_df['data_type'] = 'tweet'
    filtered_tweets_df['stock_symbol'] = CURRENT_STOCK
    logging.info(f"Filtered tweets: {len(filtered_tweets_df)}")

    # --- Processing Layer ---
    combined_df = filtered_tweets_df.copy()
    if combined_df.empty:
        logging.warning("No data to analyze. Exiting.")
        return
    combined_df.dropna(subset=['text'], inplace=True)
    logging.info(f"\n--- Combined Data for Analysis (Total: {len(combined_df)}) ---")
    logging.debug(combined_df[['data_type', 'text']].head())

    # Initialize sentiment analyzer with all models
    logging.debug("Initializing SentimentAnalyzer with models: transformer, vader, flair")
    analyzer = SentimentAnalyzer(use_cache=True, models=["transformer", "vader", "flair"])
    
    # Analyze sentiment
    logging.info("Analyzing sentiment on combined data.")
    sentiment_df = analyzer.analyze_sentiment_on_dataframe(
        combined_df.copy(), 
        text_column='text',
        content_column='content' if 'content' in combined_df.columns else None
    )
    logging.info("Sentiment analysis completed.")

    # Save the processed results
    output_file = os.path.join(output_dir, f"{CURRENT_STOCK}_sentiment_analysis_{run_timestamp}.csv")
    sentiment_df.to_csv(output_file, index=False)
    logging.info(f"Saved sentiment analysis results to {output_file}")

    # Generate summary statistics
    logging.info("Generating summary statistics.")
    sentiment_stats = {
        'stock_symbol': CURRENT_STOCK,
        'date': datetime.now().strftime("%Y-%m-%d"),
        'total_items': len(sentiment_df),
        'news_count': len(sentiment_df[sentiment_df['data_type'] == 'news']),
        'tweet_count': len(sentiment_df[sentiment_df['data_type'] == 'tweet']),
        'positive_count': len(sentiment_df[sentiment_df['sentiment_category'] == 'POSITIVE']),
        'neutral_count': len(sentiment_df[sentiment_df['sentiment_category'] == 'NEUTRAL']),
        'negative_count': len(sentiment_df[sentiment_df['sentiment_category'] == 'NEGATIVE']),
        'avg_sentiment': sentiment_df['sentiment_score'].mean(),
        'news_avg_sentiment': sentiment_df[sentiment_df['data_type'] == 'news']['sentiment_score'].mean(),
        'tweet_avg_sentiment': sentiment_df[sentiment_df['data_type'] == 'tweet']['sentiment_score'].mean() 
            if 'tweet' in sentiment_df['data_type'].values else None
    }
    logging.debug(f"Summary statistics: {sentiment_stats}")

    # Create summary DataFrame
    summary_df = pd.DataFrame([sentiment_stats])
    summary_file = os.path.join(output_dir, f"{CURRENT_STOCK}_sentiment_summary_{run_timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"Saved summary statistics to {summary_file}")

    logging.info("Process completed successfully!")

if __name__ == "__main__":
    logging.info("Starting the sentiment-driven stock prediction pipeline.")
    asyncio.run(main())
