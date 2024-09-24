import anthropic
import csv
import json
import os
import argparse
import time
from datetime import datetime
from openpyxl import Workbook
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from unidecode import unidecode
import re
import chardet
import glob
import tiktoken
import nbformat
import requests
from urllib.parse import urlparse, unquote
import snowflake.connector
from snowflake.connector.errors import ProgrammingError, DatabaseError

client = anthropic.Anthropic(
    api_key="",
)

input_dir = "/Users/navinnishanth/Downloads/BIO_TESTS/notebook/input/fortune500"
results_dir = "/Users/navinnishanth/Downloads/BIO_TESTS/notebook/results/fortune500"

MAX_TOKENS = 8000  # Increased token limit
MAX_RETRIES = 5
CHUNK_OVERLAP = 1000  # Token overlap between chunks

os.makedirs(results_dir, exist_ok=True)

log_file = os.path.join(results_dir, "generation_log.txt")
excel_filename = os.path.join(results_dir, "result.xlsx")

OKTA_USER = 'NKS'

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def extract_company_info(text):
    company_pattern = r'Company Name: (.*?)\nCompany Address:\n- Street: (.*?)\n- City: (.*?)\n- County: (.*?)\n- State: (.*?)\n- Country: (.*?)\n- ZIP: (.*?)\nCompany Revenue: (.*?)\nCompany Headcount: (.*?)\nCompany Industry: (.*?)\nNAICS Code: (.*?)\nSIC Code: (.*?)\nCompany Website: (.*?)\nCompany Website Status: (.*?)\nCompany Description: (.*?)\nCompany Phone: (.*?)\nHeadquarter Identification: (.*?)(?:\n\n|\Z)'
    match = re.search(company_pattern, text, re.DOTALL)
    if match:
        return {
            "company_name": match.group(1).strip(),
            "street": match.group(2).strip(),
            "city": match.group(3).strip(),
            "county": match.group(4).strip(),
            "state": match.group(5).strip(),
            "country": match.group(6).strip(),
            "zip": match.group(7).strip(),
            "revenue": match.group(8).strip(),
            "headcount": match.group(9).strip(),
            "industry": match.group(10).strip(),
            "naics_code": match.group(11).strip(),
            "sic_code": match.group(12).strip(),
            "website": match.group(13).strip(),
            "website_status": match.group(14).strip(),
            "description": match.group(15).strip(),
            "phone": match.group(16).strip(),
            "is_headquarter": match.group(17).strip()
        }
    return None

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def chunk_text(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), MAX_TOKENS - CHUNK_OVERLAP):
        chunk = encoding.decode(tokens[i:i + MAX_TOKENS])
        chunks.append(chunk)
    
    return chunks

def process_text_chunk(chunk, file_path, chunk_number):
    prompt_template = """
    As an AI assistant, your task is to extract comprehensive information about a specific company based on the content of this text chunk. This is chunk {chunk_number} of the file. Please gather and present any available details in a structured format:

    1. Company Name
    2. Full Company Address (including Street, City, County, State, Country, and ZIP)
    3. Company Revenue
    4. Company Headcount
    5. Company Industry
    6. NAICS Code
    7. SIC Code
    8. Company Website
    9. Company Website Status (active, inactive, etc.)
    10. Brief Company Description
    11. Company Phone Number
    12. Headquarter Identification (Yes or No)

    If complete information is not available, please provide any partial details you can find. If you cannot find any information for a field, leave it blank.

    Present the information in the following format:

    Company Name: 
    Company Address:
    - Street: 
    - City: 
    - County: 
    - State: 
    - Country: 
    - ZIP: 
    Company Revenue: 
    Company Headcount: 
    Company Industry: 
    NAICS Code: 
    SIC Code: 
    Company Website: 
    Company Website Status: 
    Company Description: 
    Company Phone: 
    Headquarter Identification: 

    Text content chunk: {content}
    """

    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt_template.format(content=chunk, chunk_number=chunk_number)}
                ]
            )
            end_time = time.time()
            
            response_text = message.content[0].text
            company_info = extract_company_info(response_text)
            
            if company_info:
                log_message(f"Extracted company info from file chunk {file_path} (chunk {chunk_number})")
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                input_cost = input_tokens * 0.25 / 1000000
                output_cost = output_tokens * 1.25 / 1000000
                total_cost = input_cost + output_cost
                time_taken = end_time - start_time
                
                log_message(f"File chunk processed: {file_path} (chunk {chunk_number}), Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Cost: ${total_cost:.2f}, Time Taken: {time_taken:.2f}s")
                
                return company_info
            else:
                log_message(f"No company info extracted from file chunk {file_path} (chunk {chunk_number})")
        except Exception as e:
            log_message(f"Error processing file chunk {file_path} (chunk {chunk_number}), attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                log_message(f"Max retries reached for file chunk {file_path} (chunk {chunk_number}). Skipping.")
                return None
            time.sleep(5)
    
    return None

def download_file(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        log_message(f"Error downloading file from {url}: {str(e)}")
        return None

def process_url(url):
    log_message(f"Processing URL: {url}")
    
    content = download_file(url)
    if not content:
        return None
    
    file_name = os.path.join(input_dir, os.path.basename(urlparse(url).path))
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(content)
    
    chunks = chunk_text(content)
    log_message(f"Created {len(chunks)} chunks for {url}")
    
    all_info = []
    for i, chunk in enumerate(chunks, start=1):
        log_message(f"Processing chunk {i}/{len(chunks)} for {url}")
        chunk_info = process_text_chunk(chunk, url, i)
        if chunk_info:
            all_info.append(chunk_info)
    
    combined_info = {}
    for info in all_info:
        for key, value in info.items():
            if value and value != "N/A":
                if key not in combined_info or len(value) > len(combined_info[key]):
                    combined_info[key] = value
    
    if combined_info:
        combined_info['source_url'] = url
    
    return combined_info if combined_info else None

def get_urls_from_snowflake():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ctx = snowflake.connector.connect(
                user=f'{OKTA_USER}',
                account='discoverorg',
                authenticator='externalbrowser',
                warehouse='WH_DATA_ENABLEMENT_LARGE',
                role='TEAM_DATA_ENABLEMENT'
            )

            cursor = ctx.cursor()
            cursor.execute("SELECT submitted_form_url FROM prd_rnd_web_acquisition.ods_sec_data.sec_archive LIMIT 1000")
            urls = [row[0] for row in cursor.fetchall()]
            ctx.close()
            return urls

        except (ProgrammingError, DatabaseError) as e:
            log_message(f"Snowflake connection attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                log_message("Max retries reached. Unable to connect to Snowflake.")
                raise
            time.sleep(5)

    return []

def process_urls(urls):
    all_company_info = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(process_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                company_info = future.result()
                if company_info:
                    all_company_info.append(company_info)
            except Exception as e:
                log_message(f"Error processing {url}: {str(e)}")
    
    log_message(f"Total companies processed: {len(all_company_info)}")
    return all_company_info

def process_text_file(file_path):
    log_message(f"Processing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = chunk_text(content)
    log_message(f"Created {len(chunks)} chunks for {file_path}")
    
    all_info = []
    for i, chunk in enumerate(chunks, start=1):
        log_message(f"Processing chunk {i}/{len(chunks)} for {file_path}")
        chunk_info = process_text_chunk(chunk, file_path, i)
        if chunk_info:
            all_info.append(chunk_info)
    
    combined_info = {}
    for info in all_info:
        for key, value in info.items():
            if value and value != "N/A":
                if key not in combined_info or len(value) > len(combined_info[key]):
                    combined_info[key] = value
    
    if combined_info:
        combined_info['source_file'] = file_path
    
    return combined_info if combined_info else None

def get_txt_files_from_input_dir():
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    log_message(f"Found {len(txt_files)} .txt files in the input directory")
    return txt_files

def process_input_files(file_paths):
    all_company_info = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_file = {executor.submit(process_text_file, file_path): file_path for file_path in file_paths}
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                company_info = future.result()
                if company_info:
                    all_company_info.append(company_info)
            except Exception as e:
                log_message(f"Error processing {file_path}: {str(e)}")
    
    log_message(f"Total companies processed from input files: {len(all_company_info)}")
    return all_company_info

if __name__ == "__main__":
    try:
        log_message("Attempting to connect to Snowflake...")
        try:
            urls = get_urls_from_snowflake()
            log_message(f"Retrieved {len(urls)} URLs from Snowflake")
        except Exception as snowflake_error:
            log_message(f"Failed to connect to Snowflake: {str(snowflake_error)}")
            urls = []

        if not urls:
            log_message("No URLs retrieved from Snowflake. Falling back to processing .txt files from input directory.")
            file_paths = get_txt_files_from_input_dir()
            if not file_paths:
                log_message("No .txt files found in the input directory. Exiting.")
                exit(1)
            all_company_info = process_input_files(file_paths)
        else:
            all_company_info = process_urls(urls)
        
        if all_company_info:
            output_df = pd.DataFrame(all_company_info)
            log_message(f"Output DataFrame shape: {output_df.shape}")
            log_message(f"Output DataFrame columns: {output_df.columns}")

            with pd.ExcelWriter(excel_filename) as writer:
                output_df.to_excel(writer, sheet_name='Processed Data', index=False)

            log_message(f"\nProcessing complete. Results saved to {excel_filename}")
        else:
            log_message("No company information was processed. Please check the input sources and try again.")
    except KeyboardInterrupt:
        log_message("Script interrupted by user. Progress has been saved.")
    except Exception as e:
        log_message(f"An error occurred: {str(e)}")
        log_message("If the error persists, please try the following:")
        log_message("1. Verify your Snowflake credentials and connection parameters.")
        log_message("2. Ensure you have the necessary permissions to access the specified database, schema, and table.")
        log_message("3. Check your network connection and firewall settings.")
        log_message("4. Update the Snowflake connector library: pip install --upgrade snowflake-connector-python")
        log_message("5. If using a VPN, try disconnecting and connecting directly.")
        log_message("6. Ensure that you have .txt files in the input directory as a fallback option.")

    # Add this section to process .txt files even if an error occurs
    if 'all_company_info' not in locals() or not all_company_info:
        log_message("Attempting to process .txt files as a final fallback.")
        file_paths = get_txt_files_from_input_dir()
        if file_paths:
            all_company_info = process_input_files(file_paths)
            if all_company_info:
                output_df = pd.DataFrame(all_company_info)
                log_message(f"Output DataFrame shape: {output_df.shape}")
                log_message(f"Output DataFrame columns: {output_df.columns}")

                with pd.ExcelWriter(excel_filename) as writer:
                    output_df.to_excel(writer, sheet_name='Processed Data', index=False)

                log_message(f"\nProcessing complete. Results saved to {excel_filename}")
            else:
                log_message("No company information was processed from .txt files.")
        else:
            log_message("No .txt files found in the input directory. Exiting.")
