# Sentiment-Driven Stock Prediction Platform

A real-time stock prediction platform that leverages sentiment analysis of news and social media combined with historical price data to forecast stock price movements and volatility.

## Project Structure

```
├── main.py                      # Main entry point for running the pipeline
├── requirements.txt             # Python dependencies
├── output/                      # Directory for storing output files
├── src/
│   ├── scraping/                # Web scraping modules
│   │   ├── news_scraper.py      # Scrapes news articles from financial websites
│   │   └── twitter_scraper.py   # Scrapes tweets related to stocks
│   └── processing/              # Data processing and sentiment analysis
│       └── sentiment_analyzer.py # Performs sentiment analysis on text data
```

## Features

### Scraping Layer
- **News Scraping**: Scrapes financial news from multiple sources (CNBC, Yahoo Finance, etc.)
  - Source-specific selectors for accurate extraction
  - Automatic stock symbol detection
  - Full article content extraction for improved sentiment analysis
  - Browser stealth mode to avoid detection

- **Twitter Scraping**: Scrapes tweets related to stocks and companies
  - Uses persistent Chrome profiles to maintain login sessions
  - Extracts rich tweet metadata (likes, retweets, etc.)
  - Implements scrolling to load more tweets
  - Handles rate limiting and detection avoidance

### Processing Layer
- **Multi-model Sentiment Analysis**: Uses three complementary NLP models:
  - **Transformer**: Deep learning model from Hugging Face
  - **VADER**: Rule-based sentiment analyzer specialized for social media
  - **Flair**: Context-aware sentiment analysis

- **Enhanced Features**:
  - Model ensemble with weighted scoring
  - Normalized sentiment scores (-1 to 1)
  - Sentiment categorization (POSITIVE, NEUTRAL, NEGATIVE)
  - Performance optimization with caching
  - Handles long text through chunking

### Data Management
- Timestamp-based file naming for historical tracking
- Summary statistics generation
- Separate directories for raw and processed data

## Usage

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create directories:
```bash
mkdir -p output
mkdir -p ~/chrome-data/twitter-profile
```

### Running

1. Run the main script:
```bash
python main.py
```

### Configuration

Edit the configuration section in `main.py` to:
- Change the target stock symbols
- Add/modify news sources
- Adjust Twitter search queries
- Configure Chrome user profile paths

## Chrome User Profiles

The Twitter scraper uses Chrome user profiles to maintain login sessions:

1. First run: Set `HEADLESS = False` to manually log in
2. After login: Set `HEADLESS = True` for automated scraping
3. Different profiles: Create separate directories for different accounts

```python
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), "chrome-data", "twitter-profile")
```

## Output Files

The system generates three types of output files:
1. Raw scraped data: `{STOCK}_news_raw_{timestamp}.csv` and `{STOCK}_tweets_raw_{timestamp}.csv`
2. Sentiment analysis results: `{STOCK}_sentiment_analysis_{timestamp}.csv`
3. Summary statistics: `{STOCK}_sentiment_summary_{timestamp}.csv`

## Future Enhancements

- Add more social media sources (Reddit, StockTwits)
- Implement entity extraction for better topic linking
- Add time series analysis for trend detection
- Integrate with historical price data for correlation analysis
