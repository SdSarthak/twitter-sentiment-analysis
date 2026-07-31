import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


import tweepy
from bs4 import BeautifulSoup
import requests
import re


import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException


import time
import os
import warnings
import random


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score

warnings.filterwarnings('ignore')

###Selenium Scraping


# Setup driver with options
options = Options()
options.add_argument("--start-maximized")  # Optional: Maximize the window
options.add_argument("--disable-extensions")  # Optional: Disable extensions

# Step 1: Set the path to your downloaded chromedriver
driver_path = os.environ["CHROMEDRIVER_PATH"]  # Point this at your local chromedriver executable

# Step 2: Set up the WebDriver Service
service = Service(driver_path)

# Step 3: Initialize ChromeDriver
driver = webdriver.Chrome(service=service, options=options)

# Example: Open Google
driver.get("https://twitter.com/login")

# Define a function to handle login
def login_to_twitter(driver, username, email, password):
    try:
        # Wait for the username field to be visible and interact with it
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "text"))
        )
        username_field.send_keys(username)
        
        # Click Next button after entering the username
        next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
        )
        next_button.click()
        
        time.sleep(2)
        # Check if email field is present, otherwise handle alternative steps
        try:
            # Check if email field is visible
            verify_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "email"))
            )
            verify_field.send_keys(email)
            
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
            )
            next_button.click()
        except:
            # If email is not found, proceed directly to password step
            print("No email verification step found, proceeding directly to password")

        # Check if password field is visible (both username and password fields might be visible at the same time)
        password_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "password"))
        )
        password_field.send_keys(password)

        # Locate and click the 'Log in' button
        login_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Log in']"))
        )
        login_button.click()

        print("Logged in successfully!")
    except Exception as e:
        print(f"An error occurred during login: {e}")
        # Optionally, handle login failure (e.g., retry, log error, etc.)

# Use the function to login
login_to_twitter(
    driver,
    os.environ["TWITTER_USERNAME"],
    os.environ["TWITTER_EMAIL"],
    os.environ["TWITTER_PASSWORD"],
)
# Wait for the login to complete
time.sleep(3)

# Step 2: Open Twitter Search Page
query = "#Budget2025"
url = f"https://x.com/search?q=%23Budget2025&src=typeahead_click"
driver.get(url)

# Allow time for page to load
time.sleep(2)


# Step 3: Scroll and Scrape Tweets


def scrape_tweets(driver, scroll_pause=2, max_scrolls=50):
    tweets_data = []  # List to store tweet content
    scroll_count = 0

    last_height = driver.execute_script("return document.body.scrollHeight")  # Get the initial scroll height

    while True:
        # Extract tweets using the provided XPath
        tweets = driver.find_elements(By.XPATH, '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[3]/section/div/div/div[8]/div[1]/div/article')

        for tweet in tweets:
            try:
                # Extract tweet text from the tweet
                tweet_text = tweet.text  # Getting the entire text of the tweet
                tweets_data.append(tweet_text)
            except Exception as e:
                continue  # Skip if there is an error extracting the tweet text
  
            # Randomly select a scroll pause time between 0 and 2 seconds
        scroll_pause = round(random.uniform(0, 2), 2)

        # Calculate new scroll height after scrolling
        new_height = driver.execute_script("return document.body.scrollHeight")
        time.sleep(scroll_pause)

        # Scroll to the bottom of the page  
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Stop if we have reached the bottom of the page (no more tweets to load)
        if new_height == last_height:
            break
        
        # Update last height to the new height after scrolling
        last_height = new_height
        scroll_count += 1
        
        # Limit number of scrolls to avoid infinite loop
        if scroll_count >= max_scrolls:
            break

    # Step 3: Save Tweets to CSV
    df = pd.DataFrame({"Tweet": tweets_data})
    df.to_csv("tweets_scraped_selenium.csv", index=False)
    print(f"Scraped {len(tweets_data)} tweets. Saved to tweets_scraped_selenium.csv")

# Step 4: Call the scrape_tweets function
scrape_tweets(driver, max_scrolls=40)