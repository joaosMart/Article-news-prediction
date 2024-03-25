# Article News Predictions

This project was done with the main goal of practicing and applying my knowledge on how to build an end-to-end machien learning project. This goes from scraping data, feature enginerring up to training and evaluating the model and deploying it in a local server using Flask and an EC2 instance. The main motivation was the practice of model deployment and scraping data in a usable format to be further used in a model. 

## Data 

The data in use was scraped from multiple websites with icelandic news. From the websites the article itself, its title and the topics of the article were scraped. Each article may have one or more topic and therefore this will be a multilabelling exercise. 

The label distribution found in the data follows the following distribution. 

![topics_distribution](images/label_articles_distribution.png)

Below one can observe the distribution of the number of labels per article. 

![number-labels-distribution](images/labels_per_article.png)

## Feature Engineering

In order to get the data in a format the model can be trained on the articles need to be processed. First the TF-IDF was applied in order to vectorize the articles. 

