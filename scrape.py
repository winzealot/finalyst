from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize, word_tokenize
from urllib.request import urlopen, Request
from transformers import pipeline
from tabulate import tabulate
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import numpy as np
import trafilatura
import sys
import re



#TODO: change the "today" logic to reference exclusively the last day's articles, rather than datetime.now's today
#TODO: implement the sentiment analysis, choose to use either FinBERT (https://huggingface.co/ProsusAI/finbert) or nltk's vader.
#TODO: create parent class, maybe called "portfolio", that contains many tickers for management
#TODO: migrate sentiment analysis to parent class over a whole portfolio rather than each ticker (for more efficiency)

def clean_text(text):
    # Basic regex cleaning
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'[\[\]{}()<>]', '', text)  # remove brackets
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\'\" \n]', '', text)  # strip special symbols
    return text.strip()

def biased_summarize(text, keyword:str, max_tokens=500):
    """
    Creates a biased summary containing strictly sentences containing keyword.
    #TODO: fix documentation here
    :param text
    :param keyword:
    :param max_tokens:
    :return:
    """
    cleaned = clean_text(text)

    sentences = sent_tokenize(cleaned)

    relevant_sents = [s for s in cleaned.split('.') if keyword.lower() in s.lower()]

    relevant_sents.extend(sentences[:10])

    summary = []
    token_count = 0
    for sent in relevant_sents:
        tokens = word_tokenize(sent)
        if token_count + len(tokens) <= max_tokens:
            summary.append(sent)
            token_count += len(tokens)
        else:
            break

    return summary

def inv_binary_summary(text, keyword:str, max_tokens=500):
    #search from the center-outwards to pick relevant sentences to the keyword, max of max_tokens

    cleaned = clean_text(text)
    sentences = sent_tokenize(cleaned)

    mid_ind = len(sentences) // 2
    left_ind = mid_ind
    right_ind = mid_ind + 1
    relevant_sents = []
    extra_sents = []
    token_count = 0
    while token_count < max_tokens:
        if left_ind >= 0 and keyword.lower() in sentences[left_ind].lower():
            relevant_sents.append(sentences[left_ind])
            token_count += len(word_tokenize(sentences[left_ind]))
        elif right_ind < len(sentences) and keyword.lower() in sentences[right_ind].lower():
            relevant_sents.append(sentences[right_ind])
            token_count += len(word_tokenize(sentences[right_ind]))
        else:
            extra_sents.append(sentences[left_ind])
            #extra_sents.append(sentences[right_ind])
        left_ind -= 1
        right_ind += 1

    while token_count < max_tokens and extra_sents:
        sent = extra_sents.pop(0)
        relevant_sents.append(sent)
        token_count += len(word_tokenize(sent))

    return relevant_sents




class TickerNewsScraper:
    def __init__(self, ticker):
        """Make a stock scraper, pulling info from FinViz
        Attributes:
            ticker: str | The stock ticker to scrape (AAPL, MSFT, etc.)
        """
        self.articles = None
        self.ticker = ticker
        self.url = f'https://finviz.com/quote.ashx?t={ticker}' if ticker else None
        #TODO: remove below + sentiment function when creating portfolio class
        self.sentimenter_one = pipeline("sentiment-analysis",
                               model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert"),
                               tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert"))

        self.sentimenter_two = pipeline("sentiment-analysis",
                                        model = AutoModelForSequenceClassification.from_pretrained(
                                            "yiyanghkust/finbert-tone"),
                                        tokenizer = AutoTokenizer.from_pretrained(
                                            "yiyanghkust/finbert-tone"))


    def pull_article_list(self, only_today=False, exclude_today=False):
        """
        Gets all the relevant articles for the current ticker.
        Leaving both only_today and exclude_today to false returns all articles.
        Having both set to true will build an empty ticker.
        :param only_today: only build ticker on most recent day's articles exclusively
        :param exclude_today: only build ticker on all days before today
        """
        if only_today and exclude_today:
            #one cannot both exclude today's info yet have only today's info
            self.articles = pd.DataFrame([])
            return

        req = Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(req)
        html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        news = soup.find_all('a', class_='tab-link-news')
        # Get the date and time of the news
        date_time = soup.find_all('td', width='130')
        self.articles = pd.DataFrame(columns=['datetime', 'title', 'link', 'text'])
        day = None
        num_elapsed = 0
        for i, item in enumerate(news):
            progress = (i + 1) / len(news) * 100
            sys.stdout.write("\r")
            sys.stdout.write("Collecting article paths: [%-20s] %d%%" % ('=' * int(progress / 5), progress))
            sys.stdout.flush()
            sleep(0.25)
            date = date_time[i].get_text().split(' ')
            #clean up, and add the day if it is not already there
            date = [x for x in date if x != '' and x != '\r\n']

            if 'Today' in date[0]:
                #it explicitly says "today"
                day = pd.to_datetime('now').strftime('%b %d %y')
            elif ':' not in date[0]:
                #it's not explicitly "today", instead it's a past date
                day = pd.to_datetime(date[0], format = '%b-%d-%y').strftime('%b %d %y')

            #now that we have the day, we must check to make sure only_today is satisfied

            if only_today and day != pd.to_datetime('now').strftime('%b %d %y'):
                #if only_today is supplemented, exclude articles from past
                continue
            if exclude_today and day == pd.to_datetime('now').strftime('%b %d %y'):
                #if exclude_today is supplemented, exclude today's articles
                continue

            if date[-1][-2:] == '\r\n':
                #there tends to be a training '\r\n' at the end of the timestamp, so gotta clean it
                date[-1] = date[-1][:-2]

            if ':' in date[0]:
                #if it's just a time stamp, we need to add the day
                date = [day] + date
            else:
                #otherwise replace existing day at date[0]
                date[0] = day
                date = [' '.join(date)]

            title = item.get_text()
            link = item['href']
            if 'http' not in link:
                link = 'https://finviz.com' + link
            temp = pd.DataFrame([[' '.join(date), title, link, '']], columns=['datetime', 'title', 'link',  'text'])
            self.articles = pd.concat([self.articles, temp], ignore_index=True)
            num_elapsed += 1
        print(f'\nRetrieved {num_elapsed} articles' + (' from today' if only_today else ''))


    def print_articles(self):
        """Prints the articles in a readable format using pandas"""
        print(tabulate(self.articles, headers='keys', showindex = False))


    def retrieve_article_contents(self):
        """
        uses Trafilatura to scrape and return the articles linked in ticker's article DataFrame.
        :return:
        """
        if self.articles is None:
            return
        for i, row in self.articles.iterrows():
            progress = (i + 1) / len(self.articles) * 100
            sys.stdout.write("\r")
            sys.stdout.write(
                "Getting the articles: [%-20s] %d%%" % ('=' * int(progress / 5), progress))
            sys.stdout.flush()
            sleep(0.25)

            if row['text'] != '':
                continue
            link = row['link']
            downloaded = trafilatura.fetch_url(link)
            if downloaded is not None:
                self.articles.at[i, 'text'] = clean_text(trafilatura.extract(downloaded))
        sys.stdout.write('\n')


    def analyze_news(self, only_today=False, exclude_today=False, keyword=None, tab=False):
        """
        Pipeline for collecting all news information and performing sentiment analysis on it.
        Note that setting only_today and exclude_today both to true will build a null ticker.
        :param only_today: only produce and analyze articles from today
        :param exclude_today: only produce and analyze articles before today
        :param tab: tabulate/print the output
        :return:
        """
        self.pull_article_list(only_today, exclude_today)
        self.retrieve_article_contents()
        self.get_sentiment(keyword)
        self.print_articles() if tab else None


    def get_sentiment(self, keyword:str=None):
        """
        TODO: fix this implementation to use vectorization (isolate the text col and apply sentiment analysis to it, trying to split the output to two diff cols
        Returns the sentiment of each article in ticker's article DataFrame.
        By default, appends sentiment as ints +1/0/-1 for positive/neutral/negative respectively
        :return: None
        """
        keyword = keyword if keyword else self.ticker
        if self.articles is None:
            return

        sentiments = []
        text:np.ndarray = self.articles['text'].to_numpy()
        sentiment_list = []
        # results = None
        # mapping = lambda x: 1 if x['label'] == 'positive' else 0 if x['label'] == 'neutral' else -1
        # for x in text:
        #     sentiment_list = np.ndarray([list(self.sentimenter_two(g).values()) for g in inv_binary_summary(x)])
        #     #aggregate results
        #     results = np.ndarray([[mapping(x[0]), x[1]] for x in sentiment_list]).mean(axis=0)
        #
        #
        #
        # #text = np.ndarray([self.sentimenter_two(g) for g in (inv_binary_summary(x) for x in text)]) #sentencewise sentiment analysis
        # text = np.ndarray([self.sentimenter_two(' '.join(inv_binary_summary(x))) for x in text]) #summary sentiment analysis
        #

        for i, row in self.articles.iterrows():
            progress = (i + 1) / len(self.articles) * 100
            sys.stdout.write("\r")
            sys.stdout.write(
                "Getting the sentiment: [%-20s] %d%%" % ('=' * int(progress / 5), progress))
            sys.stdout.flush()
            sleep(0.25)
            if row['text'] == '':
                sentiments.append([0, 0])
                continue
            sentences = biased_summarize(row['text'], keyword)
            sents = [self.sentimenter_two(x)[0] for x in sentences]
            result = max(sents, key = lambda x: x['score'])
            if result['label'] == 'positive':
                sentiments.append([1, result['score']])
            elif result['label'] == 'neutral':
                sentiments.append([0, result['score']])
            else:
                sentiments.append([-1, result['score']])

        sys.stdout.write('\n')
        self.articles = self.articles.join(pd.DataFrame(sentiments, columns=['sentiment', 'confidence']), how='inner')






