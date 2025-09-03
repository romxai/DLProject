# d:/Users/Fiona/Desktop/projects/DLProject/src/scraping/news_scraper.py
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd
import logging

# Configure logging for the scraper
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_scraper_debug.log"),
        logging.StreamHandler()
    ]
)

async def scrape_news(url: str, max_articles: int = 50, browser_args: dict = None):
    """
    Scrapes news articles from a given URL with detailed logging and timeouts.
    
    Args:
        url: The website URL to scrape news from
        max_articles: Maximum number of articles to retrieve
        browser_args: Additional arguments for browser configuration
    
    Returns:
        DataFrame with news articles and metadata
    """
    logging.info(f"Starting news scraping for URL: {url}")

    if browser_args is None:
        browser_args = {}

    # Determine the news source to customize scraping strategy
    source_type = "generic"
    if "cnbc.com" in url:
        source_type = "cnbc"
    elif "yahoo.com/finance" in url or "finance.yahoo.com" in url:
        source_type = "yahoo_finance"

    logging.debug(f"Detected source type: {source_type}")

    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
                **browser_args
            )
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            logging.info(f"Navigating to {url}")
            try:
                response = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                if not response or response.status >= 400:
                    logging.error(f"Failed to load page: {response.status if response else 'unknown error'}")
                    # Fallback: Try static HTML parsing
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    articles = []
                    for item in soup.find_all('a'):
                        title = item.get_text(strip=True)
                        link = item.get('href', '')
                        if title and link and '/news/' in link:
                            articles.append({'title': title, 'link': link, 'source': url})
                    logging.info(f"Fallback static parse found {len(articles)} articles.")
                    return pd.DataFrame(articles)
            except Exception as e:
                logging.error(f"Error during page navigation: {e}")
                # Fallback: Try static HTML parsing
                try:
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    articles = []
                    for item in soup.find_all('a'):
                        title = item.get_text(strip=True)
                        link = item.get('href', '')
                        if title and link and '/news/' in link:
                            articles.append({'title': title, 'link': link, 'source': url})
                    logging.info(f"Fallback static parse found {len(articles)} articles.")
                    return pd.DataFrame(articles)
                except Exception as e2:
                    logging.error(f"Fallback static parse also failed: {e2}")
                    return pd.DataFrame()

            logging.info("Page loaded successfully. Extracting content...")
            await page.wait_for_load_state("networkidle", timeout=30000)

            articles = []
            if source_type == "cnbc":
                try:
                    await page.wait_for_selector(".LatestNews-list", timeout=10000)
                    news_items = await page.query_selector_all(".LatestNews-item")
                    for item in news_items[:max_articles]:
                        try:
                            title_el = await item.query_selector(".LatestNews-headline")
                            link_el = await item.query_selector("a.LatestNews-headline")
                            time_el = await item.query_selector(".LatestNews-timestamp")

                            title = await title_el.inner_text() if title_el else ""
                            link = await link_el.get_attribute("href") if link_el else ""
                            timestamp = await time_el.inner_text() if time_el else ""

                            if not link.startswith('http'):
                                link = f"https://www.cnbc.com{link}"

                            articles.append({
                                'title': title,
                                'link': link,
                                'timestamp': timestamp,
                                'source': 'CNBC'
                            })
                        except Exception as e:
                            logging.warning(f"Error extracting CNBC article: {e}")
                except Exception as e:
                    logging.error(f"Error scraping CNBC articles: {e}")

            elif source_type == "yahoo_finance":
                try:
                    news_items = await page.query_selector_all('#quoteNewsStream-0-Stream li')
                    for item in news_items[:max_articles]:
                        try:
                            title_el = await item.query_selector('h3')
                            link_el = await item.query_selector('a')
                            source_el = await item.query_selector('.C(#959595)')

                            title = await title_el.inner_text() if title_el else ""
                            link = await link_el.get_attribute("href") if link_el else ""
                            source_info = await source_el.inner_text() if source_el else ""

                            if not link.startswith('http'):
                                link = f"https://finance.yahoo.com{link}"

                            articles.append({
                                'title': title,
                                'link': link,
                                'source_info': source_info,
                                'source': 'Yahoo Finance'
                            })
                        except Exception as e:
                            logging.warning(f"Error extracting Yahoo Finance article: {e}")
                except Exception as e:
                    logging.error(f"Error scraping Yahoo Finance articles: {e}")

            else:
                logging.warning(f"No specific scraping strategy for source type: {source_type}")

            logging.info(f"Scraped {len(articles)} articles from {url}")
            return pd.DataFrame(articles)

        except Exception as e:
            logging.error(f"Unexpected error during scraping: {e}")
            return pd.DataFrame()
        finally:
            await browser.close()
            logging.info("Browser closed.")

if __name__ == '__main__':
    # Example usage:
    async def main():
        news_url = "https://www.reuters.com/finance/markets" # Example URL
        news_df = await scrape_news(news_url)
        if not news_df.empty:
            print(news_df.head())
            news_df.to_csv("news_articles.csv", index=False)
            print("Saved articles to news_articles.csv")

    asyncio.run(main())
