import pandas as pd
import os
import time
import csv
import logging
import json
import argparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import threading
from queue import Queue
from tqdm import tqdm
import signal
import chardet
import anthropic
import traceback
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Initialize the Anthropic client
client = anthropic.Anthropic(
    api_key="xxx",
)

# Global variables
paused = False
pause_condition = threading.Condition()
input_file_path = 'INPUT.csv'
output_file_path = 'output.csv'
results_dir = "results/"
log_file = os.path.join(results_dir, "scraping_and_processing.log")
excel_filename = os.path.join(results_dir, "result.xlsx")

os.makedirs(results_dir, exist_ok=True)

def setup_logging():
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def log_message(message):
    logging.info(message)
    print(message)

def setup_driver():
    options = webdriver.FirefoxOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--headless')
    return webdriver.Firefox(options=options)

def scrape_content(driver, url):
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            text_content = main_content.get_text(separator=' ', strip=True)
        else:
            text_content = soup.get_text(separator=' ', strip=True)
        
        return text_content
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return ""

def extract_funding_info(text: str) -> Dict[str, Any]:
    prompt = f"""
    Given the following text, extract the funding information and return it as a JSON-formatted string:

    {text}

    The JSON should have the following keys:
    - fund_receiver: The name of the company receiving funds
    - investors: A list of investor company names
    - date: The date of the article
    - round_type: The type of funding round
    - amount_raised: The amount raised in the funding round
    - summary: A brief summary of the funding news
    - scoop_type: The type of news (e.g., Funding)
    - topics: A list of relevant topics
    - department: The business department or category

    Return only the JSON string, nothing else.
    """

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4000,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_content = response.content[0].text
    try:
        return json.loads(response_content.strip())
    except json.JSONDecodeError as e:
        log_message(f"Error parsing JSON: {str(e)}")
        log_message(f"Raw response: {response_content}")
        return {
            "error": "Failed to parse response",
            "raw_response": response_content
        }

def process_url(task_id, source_url, article_title):
    try:
        driver = setup_driver()
        content = scrape_content(driver, source_url)
        driver.quit()
        
        funding_info = extract_funding_info(content)
        
        result = {
            "TASK_ID": task_id,
            "ARTICLE TITLE": article_title,
            "SOURCE": source_url,
            "SCRAPED_CONTENT": content,
            **funding_info
        }
        
        log_message(f"Successfully processed Task ID {task_id}")
        return result
    except Exception as e:
        log_message(f"Error processing Task ID {task_id}: {e}")
        return {
            "TASK_ID": task_id,
            "ARTICLE TITLE": article_title,
            "SOURCE": source_url,
            "SCRAPED_CONTENT": "",
            "error": str(e)
        }

def worker_thread(url_queue, results, data_lock, pbar):
    while True:
        with pause_condition:
            while paused:
                pause_condition.wait()
        
        item = url_queue.get()
        if item is None:
            break
        task_id, source_url, article_title = item
        result = process_url(task_id, source_url, article_title)
        with data_lock:
            results.append(result)
        pbar.update(1)
        url_queue.task_done()

def save_partial_results(results, output_file_path):
    df = pd.DataFrame(results)
    df.to_csv(output_file_path, index=False)
    log_message(f"Partial results saved to {output_file_path}")

def pause_resume_handler(signum, frame):
    global paused
    paused = not paused
    with pause_condition:
        if paused:
            print("\nPaused. Press Ctrl+C again to resume.")
            logging.info("Processing paused")
        else:
            print("\nResuming...")
            logging.info("Processing resumed")
        pause_condition.notify_all()

def main():
    setup_logging()
    
    df = pd.read_csv(input_file_path)
    total_rows = len(df)
    
    url_queue = Queue()
    results = []
    data_lock = threading.Lock()
    num_threads = min(5, total_rows)

    signal.signal(signal.SIGINT, pause_resume_handler)

    with tqdm(total=total_rows, desc="Processing URLs") as pbar:
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker_thread, args=(url_queue, results, data_lock, pbar))
            t.start()
            threads.append(t)

        for index, row in df.iterrows():
            task_id = row['TASK_ID']
            source_url = row['SOURCE']
            article_title = row['ARTICLE TITLE']
            url_queue.put((task_id, source_url, article_title))

            if (index + 1) % 10 == 0:
                url_queue.join()
                save_partial_results(results, output_file_path)

        url_queue.join()

        for _ in range(num_threads):
            url_queue.put(None)
        for t in threads:
            t.join()

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_file_path, index=False)
    final_df.to_excel(excel_filename, index=False)
    log_message(f"Final results saved to {output_file_path} and {excel_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape and process articles data from CSV")
    parser.add_argument("--max_rows", type=int, help="Maximum number of rows to process")
    args = parser.parse_args()

    if not os.path.exists(input_file_path):
        log_message(f"Error: Input CSV file not found at {input_file_path}")
        exit(1)

    try:
        main()
    except KeyboardInterrupt:
        log_message("Script interrupted by user. Progress has been saved.")
    except Exception as e:
        log_message(f"An error occurred: {str(e)}")
        log_message(f"Traceback: {traceback.format_exc()}")
