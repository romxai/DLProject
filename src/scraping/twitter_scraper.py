# d:/Users/Fiona/Desktop/projects/DLProject/src/scraping/twitter_scraper.py
import asyncio
import logging
import os
import pandas as pd
import re
from playwright.async_api import async_playwright

# Configure logging for the scraper
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("twitter_scraper_debug.log"),
        logging.StreamHandler()
    ]
)

async def scrape_twitter(query: str, num_tweets: int = 50, user_data_dir: str = None, headless: bool = False):
    """
    Scrapes tweets for a given query using Playwright with a persistent user profile.
    This implementation is inspired by modern scraping techniques to improve reliability.
    
    Args:
        query (str): The search query for Twitter.
        num_tweets (int): Maximum number of tweets to collect.
        user_data_dir (str): Path to the Chrome user data directory for persistent login.
        headless (bool): Whether to run the browser in headless mode.
    """
    logging.info(f"Starting Twitter scrape for query: '{query}'")
    
    if not user_data_dir:
        logging.error("user_data_dir is required for persistent login.")
        return pd.DataFrame()

    tweets = []
    # Use 'live' for real-time results and encode the query
    from urllib.parse import quote
    encoded_query = quote(query)
    url = f"https://twitter.com/search?q={encoded_query}&src=typed_query&f=live"

    context = None
    try:
        async with async_playwright() as p:
            # Use launch_persistent_context to correctly handle user data directory
            context = await p.chromium.launch_persistent_context(
                user_data_dir,
                headless=headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--start-maximized" # Helps mimic real user behavior
                ],
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080} # Use a common desktop resolution
            )
            
            page = await context.new_page()
            
            # Add stealth measures
            await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            logging.info(f"Navigating to Twitter search URL: {url}")
            await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            
            # Wait for the main timeline to appear
            try:
                await page.wait_for_selector('div[data-testid="primaryColumn"]', timeout=30000)
                logging.info("Twitter timeline loaded.")
            except Exception as e:
                logging.error(f"Timeline did not load: {e}. The page might require login or be showing a captcha.")
                await context.close()
                return pd.DataFrame()

            # Scroll and scrape loop
            collected_tweets_text = set() # Use a set to store tweet text to avoid duplicates
            last_height = 0
            scroll_attempts = 0
            max_scroll_attempts = 15 # Stop if we can't find new tweets after this many scrolls

            while len(tweets) < num_tweets and scroll_attempts < max_scroll_attempts:
                await page.wait_for_timeout(2500) # Wait for content to load after scroll
                
                tweet_elements = await page.query_selector_all('article[data-testid="tweet"]')
                
                if not tweet_elements:
                    logging.warning("No tweet elements found on the page.")
                    scroll_attempts += 1
                    continue

                for tweet_element in tweet_elements:
                    try:
                        text_element = await tweet_element.query_selector('div[data-testid="tweetText"]')
                        text = await text_element.inner_text() if text_element else ""
                        # Extract all @mentions from the tweet text
                        mentions = re.findall(r'@\w+', text)

                        if text and text not in collected_tweets_text:
                            collected_tweets_text.add(text)
                            
                            user_element = await tweet_element.query_selector('div[data-testid="User-Name"]')
                            user_text = await user_element.inner_text() if user_element else "Unknown\n@Unknown"
                            user_parts = user_text.split('\n')
                            
                            user = user_parts[0]
                            handle = user_parts[1] if len(user_parts) > 1 else "@unknown"
                            
                            timestamp = ""
                            time_element = await tweet_element.query_selector('time')
                            if time_element:
                                timestamp = await time_element.get_attribute('datetime')

                            tweets.append({
                                'user': user,
                                'handle': handle,
                                'timestamp': timestamp,
                                'text': text,
                                'mentions': ','.join(mentions),
                                'source': 'Twitter',
                                'query': query
                            })
                            
                            if len(tweets) >= num_tweets:
                                break
                    except Exception as e:
                        logging.debug(f"Could not process a tweet element: {e}")

                if len(tweets) >= num_tweets:
                    logging.info(f"Collected target number of tweets ({num_tweets}).")
                    break

                logging.info(f"Collected {len(tweets)}/{num_tweets} tweets. Scrolling to load more.")
                await page.evaluate('window.scrollBy(0, window.innerHeight * 2)')
                
                # Check if scroll position has changed
                new_height = await page.evaluate('document.body.scrollHeight')
                if new_height == last_height:
                    scroll_attempts += 1
                    logging.warning(f"Scroll position did not change. Attempt {scroll_attempts}/{max_scroll_attempts}.")
                else:
                    last_height = new_height
                    scroll_attempts = 0 # Reset attempts on successful scroll

            logging.info(f"Finished scraping. Found {len(tweets)} unique tweets.")
            return pd.DataFrame(tweets)

    except Exception as e:
        logging.error(f"An unexpected error occurred in scrape_twitter: {e}", exc_info=True)
        return pd.DataFrame()
    
    finally:
        if context:
            try:
                await context.close()
                logging.info("Browser context closed.")
            except Exception as e:
                logging.warning(f"Error closing browser context: {e}")

if __name__ == '__main__':
    # Example usage:
    async def main():
        # IMPORTANT: Make sure this path exists and is writable
        user_data_path = os.path.join(os.path.expanduser("~"), "chrome-data", "twitter-profile")
        os.makedirs(user_data_path, exist_ok=True)
        
        search_query = "$TSLA" # Example stock symbol
        tweets_df = await scrape_twitter(
            search_query, 
            num_tweets=20,
            user_data_dir=user_data_path,
            headless=False # Set to False for the first run to log in
        )
        if not tweets_df.empty:
            print(tweets_df.head())
            tweets_df.to_csv("twitter_feed.csv", index=False)
            print("Saved tweets to twitter_feed.csv")

    asyncio.run(main())
