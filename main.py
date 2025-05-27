#TODO: store all collected and scraped news articles in a DB

from scrape import TickerNewsScraper


msft = TickerNewsScraper('MSFT')
msft.analyze_news(only_today=False, tab=True)

