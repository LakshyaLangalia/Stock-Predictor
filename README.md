Takes the previous stock price information and attempts to generate a prediction for future stock prices using random forests, taking numerous basic and advanced technical parameters. 
The accuracy of the model ranges between 52-65% for most stocks. Still working on implementing macroeconomic data and sentiment analysis into the model. Stocks must be listed for over
10 years for an accurate analysis. 

Takes a CSV of stock symbols and attempts to generate a prediction for which prices will go up based on the most recent information using random forests, taking numerous basic
and advanced technical parameters as well as a modified version of the sentiment analysis I've also developed. Outputs the probabilities in ascending order. 
