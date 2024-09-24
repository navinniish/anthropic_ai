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

client = anthropic.Anthropic(
    api_key="",
)

csv_file = "/Users/navinnishanth/Downloads/BIO_TESTS/Dir+ Contacts_Sample.csv"
results_dir = "/Users/navinnishanth/Downloads/BIO_TESTS/GPT_input_3_Companies/bio_results_2"

os.makedirs(results_dir, exist_ok=True)

log_file = os.path.join(results_dir, "generation_log.txt")

excel_filename = os.path.join(results_dir, "result.xlsx")

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def check_csv_contents(csv_file):
    try:
        encoding = detect_encoding(csv_file)
        log_message(f"Detected encoding: {encoding}")
        df = pd.read_csv(csv_file, encoding=encoding, nrows=5)
        log_message(f"CSV file preview (comma delimiter):\n{df.to_string()}")
        log_message(f"Column names: {df.columns.tolist()}")
        log_message(f"Data types: {df.dtypes}")
    except Exception as e:
        log_message(f"Error reading CSV file: {str(e)}")

def extract_contact_info(text, company_rank, original_data):
    contact_pattern = r'Name: (.*?)\nIndividual ID: (.*?)\nPrimary Title: (.*?)\nManagement Level: (.*?)\nEmail Address: (.*?)\nBest Freemail: (.*?)\nPhone Number: (.*?)\nLinkedIn URL: (.*?)\nCompany ID: (.*?)\nReason: (.*?)\nInfo Count: (.*?)\nContact Rank: (.*?)\nConfidence Score: (.*?)(?:\n\n|\Z)'
    matches = re.findall(contact_pattern, text, re.DOTALL)
    log_message(f"Number of matches found: {len(matches)}")
    contacts = []
    for i, match in enumerate(matches):
        try:
            individual_id = match[1] if len(match) > 1 else 'N/A'
            original_record = next((r for r in original_data if str(r['INDIVIDUAL_ID']) == individual_id), None)
            
            confidence_score = float(match[13]) if len(match) > 13 and match[13].strip() else (float(original_record['CONFIDENCE_SCORE']) if original_record and 'CONFIDENCE_SCORE' in original_record else 0)
            
            contact = {
                "name": match[0] if len(match) > 0 else 'N/A',
                "individual_id": individual_id,
                "primary_title": match[2] if len(match) > 2 else 'N/A',
                "management_level": match[3] if len(match) > 3 else 'N/A',
                "email_address": match[4] if len(match) > 4 else '',
                "best_freemail": match[5] if len(match) > 5 else '',
                "phone_number": match[6] if len(match) > 6 else '',
                "linkedin_url": match[7] if len(match) > 7 else '',
                "company_id": match[8] if len(match) > 8 else 'N/A',
                "reason": match[9] if len(match) > 9 else '',
                "info_count": int(match[10]) if len(match) > 10 and match[10].isdigit() else 0,
                "contact_rank": min(int(match[11]) if len(match) > 11 and match[11].isdigit() else 0, 5),
                "company_rank": company_rank,
                "confidence_score": confidence_score
            }
            contacts.append(contact)
            log_message(f"Processed contact: {contact}")
        except Exception as e:
            log_message(f"Warning: Issue processing contact {i}: {str(e)}")
            log_message(f"Problematic match: {match}")
            if original_record:
                fallback_contact = create_contact_from_original(original_record, company_rank, i+1)
                contacts.append(fallback_contact)
                log_message(f"Added fallback contact: {fallback_contact}")
    
    # If no contacts were extracted, use the original data
    if not contacts:
        log_message(f"No contacts extracted from API. Using original data for all contacts.")
        contacts = [create_contact_from_original(record, company_rank, i+1) for i, record in enumerate(original_data[:5])]
    
    return contacts

def create_contact_from_original(record, company_rank, contact_rank):
    return {
        "name": record.get('NAME', 'N/A'),
        "individual_id": record.get('INDIVIDUAL_ID', 'N/A'),
        "primary_title": record.get('PRIMARY_TITLE', 'N/A'),
        "management_level": record.get('MANAGEMENT_LEVEL', 'N/A'),
        "email_address": record.get('EMAIL_ADDRESS', ''),
        "best_freemail": record.get('BEST_FREEMAIL', ''),
        "phone_number": record.get('PHONE_NUMBER', ''),
        "linkedin_url": record.get('LINKEDIN_URL', ''),
        "company_id": record.get('COMPANY_ID', 'N/A'),
        "reason": "Selected based on original data",
        "info_count": sum(1 for field in ['EMAIL_ADDRESS', 'BEST_FREEMAIL', 'MOBILE_PHONE', 'PHONE_NUMBER', 'LINKEDIN_URL'] if field in record and record.get(field)),
        "contact_rank": min(contact_rank, 5),
        "company_rank": company_rank,
        "confidence_score": float(record.get('CONFIDENCE_SCORE', 0))
    }

def process_data(data, batch_size=100):
    all_contacts = []
    log_message(f"Total number of records to process: {len(data)}")
    
    # Group data by company
    company_data = {}
    for record in data:
        company_id = record['COMPANY_ID']
        if company_id not in company_data:
            company_data[company_id] = []
        company_data[company_id].append(record)
    
    # Sort companies by their average confidence score
    sorted_companies = sorted(company_data.items(), 
                              key=lambda x: sum(float(r['CONFIDENCE_SCORE']) for r in x[1]) / len(x[1]), 
                              reverse=True)
    
    for company_rank, (company_id, company_records) in enumerate(sorted_companies, start=1):
        log_message(f"Processing company ID: {company_id}, Rank: {company_rank}")
        company_contacts = []
        for i in range(0, len(company_records), batch_size):
            batch = company_records[i:i+batch_size]
            log_message(f"Processing batch for company {company_id}, size: {len(batch)}")
            prompt_template = """
        As an AI assistant, your task is select the top 5 optimal contacts from each company for Hotel Engine's sales team to approach. I have a dataset of contacts from different companies. The objective is to select the 5 ideal contacts per company who are most likely decision-makers or influential in purchasing or managing lodging solutions. The company to focus on is Hotel Engine, a lodging management software company. Hotel Engine serves industries like construction, transportation & logistics, oil & gas, retail, healthcare, and more. Their product helps with customized hotel reservation & management, billing, control, and support for both large and small businesses.
        Please identify contacts based on their job titles that are most relevant to hotel management, travel services, procurement, logistics, or operations in each company. Prioritize senior executives or managers in roles such as Travel Manager, Procurement Manager, Operations Manager, Facilities Manager, or Logistics Head, who are likely responsible for lodging or travel arrangements within their companies
        The dataset includes the following columns: INDIVIDUAL_ID, NAME, LTE_FLAG, PRIMARY_TITLE, MANAGEMENT_LEVEL, EMAIL_ADDRESS, BEST_FREEMAIL, MOBILE_PHONE, PHONE_NUMBER, LINKEDIN_URL, COMPANY_ID, and CONFIDENCE_SCORE.
        Keep in mind the following pointers while selecting the contacts:
        1. Do not select duplicate entries based on NAME and COMPANY_ID.
        2. Prefer contacts with the most relevant and influential job titles that suit the requirements, followed by the Higher CONFIDENCE_SCORE
        3. For contacts with same C-level positions in a company, retain only the entry with the highest CONFIDENCE_SCORE. For eg if person A has "chairman and ceo" and person
        B has "chief executive officer & chief financial officer", then select  one contact having higher confidence score
        4. RANK THE TOP 5 CONTACTS PER EACH COMPANY_ID FROM 1 TO 5 I.E THE RANK SHOULDN'T EXCEED 5 FOR ANY GIVEN CONTACT
        5. I definitely need five contacts per company, Incase if you cant select 5 contacts for a company, please pick the contacts based on their 
        management_level (in this order c-level , vp-level, director, manager, non manager)  and confidence_score (highest score) and it should not pick
        titles like 'Former','Retired','Resigned','Past','independent','Self Employed','Selfemploy','Unemployed','Freelance', 
        Advisor, Consultant, Personal Assistant, PA,
        Chief of Staff, Office of, to the, Secretary, Office,
        and order the final 5 contacts for each company by c-level, vp-level, director, manager, non manager and confidence_score
        6. For each selected contact, provide the following information in this exact format below:
        Name: [Contact Name]
        Individual ID: [Id of an Individual]
        Primary Title: [Contact Title]
        Management Level: [Management Level]
        Email Address: [contact@email_domain.com]
        Best Freemail: [contact@freemail_domain.com]
        Phone Number: [123-456-7890]
        LinkedIn URL: [https://linkedin.com/in/contact]
        Company ID: [12345]
        Reason: [Explanation for selection of Contact]
        Info Count: [Number of non-empty fields among EMAIL_ADDRESS, BEST_FREEMAIL, MOBILE_PHONE, PHONE_NUMBER, and LINKEDIN_URL]
        Contact Rank: [Rank within company, from 1 to 5]
        Confidence Score: [confidence_score]
        Repeat this format for each contact, separating them with a blank line.
        Data to process: {data}
            """

            try:
                start_time = time.time()
                message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt_template.format(data=json.dumps(batch))}
                    ]
                )
                end_time = time.time()
                
                response_text = message.content[0].text
                log_message(f"API Response for company {company_id} (first 500 characters): {response_text[:500]}")
                
                batch_contacts = extract_contact_info(response_text, company_rank, batch)
                log_message(f"Extracted contacts from batch for company {company_id}: {len(batch_contacts)}")
                
                if not batch_contacts:
                    log_message(f"No contacts extracted from API for company {company_id}. Using original data.")
                    batch_contacts = [create_contact_from_original(record, company_rank, i) for i, record in enumerate(batch[:5], start=1)]
                
                company_contacts.extend(batch_contacts)
                
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                input_cost = input_tokens * 0.25 / 1000000
                output_cost = output_tokens * 1.25 / 1000000
                total_cost = input_cost + output_cost
                time_taken = end_time - start_time
                
                log_message(f"Batch processed for company {company_id}: {i}-{i+batch_size}, Contacts: {len(batch_contacts)}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Cost: ${total_cost:.2f}, Time Taken: {time_taken:.2f}s")
            except Exception as e:
                log_message(f"Error processing data for company {company_id}: {str(e)}")
                log_message(f"Using original data for this batch.")
                batch_contacts = [create_contact_from_original(record, company_rank, i) for i, record in enumerate(batch[:5], start=1)]
                company_contacts.extend(batch_contacts)
        
        # Sort the company contacts by CONFIDENCE_SCORE in descending order
        company_contacts.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Select only the top 5 contacts for this company and assign ranks 1-5
        top_contacts = company_contacts[:5]
        for i, contact in enumerate(top_contacts, start=1):
            contact['contact_rank'] = i
        
        # Add the top 5 contacts from this company to the final list
        all_contacts.extend(top_contacts)
    
    log_message(f"Total contacts extracted: {len(all_contacts)}")
    return {"contacts": all_contacts}

def validate_row(row):
    required_keys = ['INDIVIDUAL_ID', 'NAME', 'PRIMARY_TITLE', 'COMPANY_ID', 'CONFIDENCE_SCORE', 'MANAGEMENT_LEVEL']
    for key in required_keys:
        if key not in row.index or pd.isna(row[key]) or row[key] == '':
            row[key] = 'N/A'
    
    for key in row.index:
        if isinstance(row[key], str):
            row[key] = unidecode(row[key])
    
    if all(row[key] == 'N/A' for key in required_keys):
        log_message(f"Warning: All required fields are 'N/A' for row: {row.to_dict()}")
    
    return row

def process_csv(csv_file, max_rows=None):
    if os.path.exists(excel_filename):
        os.remove(excel_filename)
        log_message(f"Deleted existing output file: {excel_filename}")

    check_csv_contents(csv_file)

    try:
        encoding = detect_encoding(csv_file)
        df = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
        log_message(f"CSV file read successfully with {encoding} encoding. Shape: {df.shape}")
        log_message(f"Columns: {df.columns.tolist()}")
        log_message(f"First row of data: {df.iloc[0].to_dict()}")
        
        required_columns = ['INDIVIDUAL_ID', 'NAME', 'PRIMARY_TITLE', 'COMPANY_ID', 'CONFIDENCE_SCORE', 'MANAGEMENT_LEVEL']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            log_message(f"Warning: The following required columns are missing: {missing_columns}")
        
        # Remove rows where all required fields are empty or NaN
        df = df.dropna(subset=required_columns, how='all')
        log_message(f"Rows after removing empty entries: {len(df)}")
        
        # Convert CONFIDENCE_SCORE to numeric, replacing non-numeric values with NaN
        df['CONFIDENCE_SCORE'] = pd.to_numeric(df['CONFIDENCE_SCORE'], errors='coerce')
        
        # Remove rows with NaN CONFIDENCE_SCORE
        df = df.dropna(subset=['CONFIDENCE_SCORE'])
        log_message(f"Rows after removing invalid CONFIDENCE_SCORE: {len(df)}")
        
        log_message(f"DataFrame shape before processing: {df.shape}")
        log_message(f"DataFrame head before processing:\n{df.head().to_string()}")
        
    except Exception as e:
        log_message(f"Error reading CSV file: {str(e)}")
        return

    if max_rows:
        df = df.iloc[:max_rows]

    df = df.apply(validate_row, axis=1)

    processed_data = process_data(df.to_dict('records'))

    log_message(f"Processed data: {processed_data}")

    if processed_data and 'contacts' in processed_data and isinstance(processed_data['contacts'], list) and len(processed_data['contacts']) > 0:
        log_message(f"Number of contacts in processed data: {len(processed_data['contacts'])}")
        output_df = pd.DataFrame(processed_data['contacts'])
        log_message(f"Output DataFrame:\n{output_df.head().to_string()}")

        # Ensure the required columns are present
        required_columns = ['name', 'individual_id', 'primary_title', 'management_level', 'email_address', 'best_freemail', 'phone_number', 'linkedin_url', 'company_id', 'reason', 'info_count', 'contact_rank', 'company_rank', 'confidence_score']
        for col in required_columns:
            if col not in output_df.columns:
                output_df[col] = 'N/A'

        with pd.ExcelWriter(excel_filename) as writer:
            output_df.to_excel(writer, sheet_name='Processed Data', index=False, columns=required_columns)

        log_message(f"\nProcessing complete. Results saved to {excel_filename}")
    else:
        log_message("Error: No processed data returned or invalid data structure.")
        log_message(f"Processed data structure: {type(processed_data)}")
        log_message(f"Processed data content: {processed_data}")
        # Save the original data to Excel as a fallback
        df.to_excel(excel_filename, index=False)
        log_message(f"Original data saved to {excel_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process contact data from CSV")
    parser.add_argument("--max_rows", type=int, help="Maximum number of rows to process")
    args = parser.parse_args()

    if not os.path.exists(csv_file):
        log_message(f"Error: CSV file not found at {csv_file}")
        exit(1)

    try:
        check_csv_contents(csv_file)
        process_csv(csv_file, max_rows=args.max_rows)
    except KeyboardInterrupt:
        log_message("Script interrupted by user. Progress has been saved.")
    except Exception as e:
        log_message(f"An error occurred: {str(e)}")
