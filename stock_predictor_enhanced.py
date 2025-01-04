import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from ta import add_all_ta_features
from ta.utils import dropna
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from transformers import pipeline
from datetime import datetime

# define the model -- params can be changed for deeper learning if necessary
model = RFC(n_estimators=100, min_samples_split=100, random_state=1)

# list of parameters which involve technical analysis, time of year (e.g. sales are
# higher in december, and sentiment analysis using finviz.)
predictors = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'momentum_rsi', 'trend_macd',
    'volatility_bbm', 'volatility_bbl', 'Stochastic %K', 'Stochastic %D', 
    'Bollinger MAVG', 'Bollinger High', 'Bollinger Low', 'OBV', 
    'Day of Week', 'Month', 'Quarter', 'Sentiment Difference'
]

# Sentiment analysis function
def analyze_sentiment(ticker):
    # grabs the url and appends the ticker, and sends a request to that url
    # using the appropriate headers
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    req = Request(url, headers={ 
        "User-Agent": "Mozilla/5.0"
    })
    # opens the url and parses the text, then searches for the news table, which
    # is the part of the finviz link which contains all the headlines
    response = urlopen(req)
    html = BeautifulSoup(response, 'html.parser')
    news_table = html.find(id='news-table')

    # finds all 'tr' tags and grabs the ticker, date, time and title that the article was posted
    # puts that in a list and iterates thru all tr tags.
    parsed_data = []
    for row in news_table.findAll('tr'):
        # the a tag inside each row contains the title so we extract that, and only get the text
        # not any other html identifiers
        title = row.a.text
        # the td tag contains the date and time that the article was published
        # the .split() function splits the text into a list where the first element represents the
        # date and the second represents the time

        # in some cases, the date may not appear, in that case the article was published today so 
        # add a placeholder called today which we will deal with later.
        date_data = row.td.text.split()
        date = date_data[0] if len(date_data) > 1 else 'Today'
        time = date_data[-1]
        parsed_data.append([ticker, date, time, title])

    # since every list will have 4 elements, we can create a dataframe with 4 columns
    # and replace the 'Today' placeholder with the current date and time and convert every
    # date and time to a datetime object.
    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
    df['date'] = df['date'].replace('Today', datetime.now().strftime('%b-%d-%y'))
    df['date'] = pd.to_datetime(df['date']).dt.date

    # defining the sentiment model and using finbert, which is a sentiment analyzer
    # specifically developed for finance
    sentiment_model = pipeline('text-classification', model='ProsusAI/finbert')
    # applies the function to the title column, and calls that column 'score'
    df['score'] = df['title'].apply(lambda title: sentiment_model(title)[0]['score'])
    # group by date and get the daily score
    daily_sentiment = df.groupby('date')['score'].mean()
    return daily_sentiment

def predict_today(stock_data, model, predictors):
    # grabs the most recent data for the stock
    latest_data = stock_data.iloc[-1:].copy() 

    # trains the model on the predictors and the next-day-greater column as the target, 
    # which is defined in the next function
    model.fit(stock_data[predictors], stock_data['Next Day Greater?'])

    # predicts the probability of the stock rising on the next day and returns the probability
    # of today's prediction
    probabilities = model.predict_proba(latest_data[predictors])[:, 1] 
    return probabilities[0] 

# Function to process a single stock for today's prediction
def process_stock_for_today(ticker):
    try:
        # grabs the infomration for the stock since 2000 for old stocks, 
        # otherwise grab the entire history of thes tock
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start="2000-01-01")

        if stock_data.empty:
            stock_data = stock.history(period="max")

        # drop unnecessary columns if they exist
        columns_to_drop = ['Dividends', 'Stock Splits']
        stock_data.drop(columns=[col for col in columns_to_drop if col in stock_data.columns], inplace=True)

        # drop all null values, and add all technical analysis features to the df
        stock_data = dropna(stock_data)
        stock_data = add_all_ta_features(
            stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )

        # grab the stochastic, bollinger, obv values and create a df that contains along with day of week information
        # merges that df with the original df using pd.concat
        stochastic = StochasticOscillator(
            high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], window=14, smooth_window=3
        )
        bollinger = BollingerBands(close=stock_data['Close'], window=20, window_dev=2)
        obv = OnBalanceVolumeIndicator(close=stock_data['Close'], volume=stock_data['Volume'])

        additional_columns = pd.DataFrame({
            'Stochastic %K': stochastic.stoch(),
            'Stochastic %D': stochastic.stoch_signal(),
            'Bollinger MAVG': bollinger.bollinger_mavg(),
            'Bollinger High': bollinger.bollinger_hband(),
            'Bollinger Low': bollinger.bollinger_lband(),  # Corrected here
            'OBV': obv.on_balance_volume(),
            'Day of Week': stock_data.index.dayofweek,
            'Month': stock_data.index.month,
            'Quarter': stock_data.index.quarter
        }, index=stock_data.index)

        stock_data = pd.concat([stock_data, additional_columns], axis=1)

        # creates a column called next day, which simply shifts the close price one day forward
        # then creates a new column which determines if the price was greater or not, 0 for false, 1 for true
        stock_data['Next Day'] = stock_data['Close'].shift(-1)
        stock_data['Next Day Greater?'] = (stock_data['Next Day'] > stock_data['Close']).astype(int)
        stock_data.dropna(inplace=True)

        stock_data['Trade Date'] = stock_data.index.date

        # Merge with sentiment data
        sentiment_data = analyze_sentiment(ticker)
        stock_data = stock_data.merge(sentiment_data.rename('Sentiment'), left_on='Trade Date', right_index=True, how='left')

        # uses the predict_today fcn defined above and returns the probabiltiy
        probability = predict_today(stock_data, model, predictors)
        return probability
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


# creates a dictionary with todays predictions given a series of tickers
def predict_stocks_today(stock_symbols):
    predictions_today = {}

    for ticker in stock_symbols:
        probability = process_stock_for_today(ticker)
        if probability is not None:
            predictions_today[ticker] = probability

    # sorts the predictions from ascending to descending
    sorted_predictions = sorted(predictions_today.items(), key=lambda x: x[1], reverse=True)
    return sorted_predictions

stock_symbols = pd.read_csv('stock_symbols.csv')['Symbol']
today_predictions = predict_stocks_today(stock_symbols)

print("Predictions for Today's Stock Movement (Highest to Lowest Probability):")
for ticker, probability in today_predictions:
    print(f"{ticker}: {probability:.5f}")
