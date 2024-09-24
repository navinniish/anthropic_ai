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

# Initialize the Anthropic client
client = anthropic.Anthropic(
    api_key="",
)

# Define directories
csv_file = "/Users/navinnishanth/Downloads/BIO_TESTS/bios pr.csv"
results_dir = "/Users/navinnishanth/Downloads/BIO_TESTS/bio_results"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Create a log file
log_file = os.path.join(results_dir, "generation_log.txt")

# Excel file name
excel_filename = os.path.join(results_dir, "bio_data.xlsx")

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def check_csv_contents(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding='utf-8', nrows=5)
        log_message(f"CSV file preview (comma delimiter):\n{df.to_string()}")
        log_message(f"Column names: {df.columns.tolist()}")
        log_message(f"Data types: {df.dtypes}")
    except Exception as e:
        log_message(f"Error reading CSV file: {str(e)}")

def generate_bio(data, min_length, attempt=1):
    log_message(f"Generating bio for data: {data} (Attempt: {attempt}, Min Length: {min_length})")
    
    # Check if all fields are 'N/A'
    if all(value == 'N/A' for value in data.values()):
        return {
            "name": "N/A",
            "profile_id": "N/A",
            "bio": "Insufficient data provided to generate a biography.",
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0,
            "time_taken": 0
        }

    prompt = f"""
You are an AI assistant tasked with generating professional biographies for
contacts in a specific format. Your task is to create a concise, informative biography for each contact using the following guidelines:

1. Start with the contact's full name.
2. Include their current job title and company, using one of these phrases: "at", "with", "of", or "from".
3. If location is available, include it using "based in".
4. If applicable, use "responsible for" to describe their role or responsibilities.
5. Include information about their previous position if available.
6. Mention their educational background if provided, only including the bachelor's degree or master's if he has any and college name.

The biography should be detailed and informative, highlighting the most relevant information about the person's current role, based in location, current responsibilities,
previous experience (if available . Ignore if not available),
and education. The bio must be at least {min_length} characters long.

For ex:
For persons with current position available:
{"Name"} is {"person's current position at current company"}, based in {"current location"}. He is responsible for {"current role's responsibilities"} .
Prior to his current role, {"Name"} served as {"Previous Role"} at {"Previous Company"}.
{"Name"} holds a {"Education Degree"} from {"College/Institution"}.
For persons with current position not available:
{"Name"} was {"person's most recent position at the most recent company"}, based in {"current location"}. He was responsible for {"most recent role's responsibilities"} .
{"Name"} holds a {"Education Degree"} from {"College/Institution"}.

Name: {data.get('FULL_NAME', 'N/A')}
Location: {data.get('LOCATION', 'N/A')}
Current Company: {data.get('COMPANY_NAME', 'N/A')}
Current Position: {data.get('CURRENT_POSITION', 'N/A')}
Previous Company: {data.get('COMPANY_NAME_PREV', 'N/A')}
Previous Position: {data.get('PREVIOUS_POSITION', 'N/A')}
Degree: {data.get('DEGREE', 'N/A')}
Institution: {data.get('INSTITUTION_NAME', 'N/A')}
Social URL: {data.get('SOCIAL_URL', 'N/A')}

Please write a detailed professional bio for this person following the guidelines provided above.
Include specific details about their current role, previous experience, and education based on the information provided.

This is attempt number {attempt}. Please ensure the bio is at least {min_length} characters long.

Return the bio as a JSON string with the following structure:
{{
  "name": "{data.get('FULL_NAME', 'N/A')}",
  "profile_id": "{data.get('PROFILE_ID', 'N/A')}",
  "bio": "GENERATED_BIO"
}}

Please return ONLY the JSON string as described above, with no additional text before or after.
"""

    try:
        start_time = time.time()
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4000,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        end_time = time.time()
        
        json_match = json.loads(message.content[0].text)
        
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        input_cost = input_tokens * 0.25 / 1000000  # $0.25 per million tokens
        output_cost = output_tokens * 1.25 / 1000000  # $1.25 per million tokens
        total_cost = input_cost + output_cost
        time_taken = end_time - start_time
        
        json_match["input_tokens"] = input_tokens
        json_match["output_tokens"] = output_tokens
        json_match["input_cost"] = input_cost
        json_match["output_cost"] = output_cost
        json_match["total_cost"] = total_cost
        json_match["time_taken"] = time_taken
        
        return json_match
    except Exception as e:
        log_message(f"Error generating bio for {data.get('FULL_NAME', 'Unknown')}: {str(e)}")
        return None

def evaluate_bio(name, bio):
    prompt = f"""
Please evaluate the following bio for {name} on a scale of 1-10 for quality and accuracy, where 1 is very poor and 10 is excellent. Provide a brief explanation for your rating.

Bio: {bio}

Return your evaluation as a JSON string with the following structure:
{{
  "name": "{name}",
  "rating": NUMERIC_RATING,
  "explanation": "YOUR_EXPLANATION"
}}

Please return ONLY the JSON string as described above, with no additional text before or after.
"""

    try:
        start_time = time.time()
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4000,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        end_time = time.time()
        
        json_match = json.loads(message.content[0].text)
        
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        input_cost = input_tokens * 0.25 / 1000000  # $0.25 per million tokens
        output_cost = output_tokens * 1.25 / 1000000  # $1.25 per million tokens
        total_cost = input_cost + output_cost
        time_taken = end_time - start_time
        
        json_match["input_tokens"] = input_tokens
        json_match["output_tokens"] = output_tokens
        json_match["input_cost"] = input_cost
        json_match["output_cost"] = output_cost
        json_match["total_cost"] = total_cost
        json_match["time_taken"] = time_taken
        
        return json_match
    except Exception as e:
        log_message(f"Error evaluating bio for {name}: {str(e)}")
        return None

def validate_row(row):
    required_keys = ['LOCATION', 'FULL_NAME', 'COMPANY_NAME', 'CURRENT_POSITION', 'PROFILE_ID', 'PERSON_BIOGRAPHY']
    for key in required_keys:
        if key not in row.index or pd.isna(row[key]) or row[key] == '':
            row[key] = 'N/A'
    
    # Check if all required fields are 'N/A'
    if all(row[key] == 'N/A' for key in required_keys):
        log_message(f"Warning: All required fields are 'N/A' for row: {row.to_dict()}")
    
    return row

def process_profile(row):
    try:
        log_message(f"Processing profile: {row.get('FULL_NAME', 'Unknown')}")
        log_message(f"Profile data: {row}")
        
        person_bio_length = len(row.get('PERSON_BIOGRAPHY', ''))
        min_length = max(person_bio_length, 200)  # Ensure a minimum length of 200 characters
        attempt = 1
        max_attempts = 5  # Maximum number of attempts to generate a longer bio
        
        while attempt <= max_attempts:
            bio_data = generate_bio(row, min_length, attempt)
            if bio_data:
                ai_bio_length = len(bio_data['bio'])
                if ai_bio_length >= min_length:
                    evaluation = evaluate_bio(bio_data['name'], bio_data['bio'])
                    if evaluation:
                        return {
                            'name': bio_data['name'],
                            'profile_id': bio_data['profile_id'],
                            'bio': bio_data['bio'],
                            'rating': evaluation['rating'],
                            'explanation': evaluation['explanation'],
                            'bio_input_tokens': bio_data['input_tokens'],
                            'bio_output_tokens': bio_data['output_tokens'],
                            'bio_input_cost': bio_data['input_cost'],
                            'bio_output_cost': bio_data['output_cost'],
                            'bio_total_cost': bio_data['total_cost'],
                            'bio_time_taken': bio_data['time_taken'],
                            'eval_input_tokens': evaluation['input_tokens'],
                            'eval_output_tokens': evaluation['output_tokens'],
                            'eval_input_cost': evaluation['input_cost'],
                            'eval_output_cost': evaluation['output_cost'],
                            'eval_total_cost': evaluation['total_cost'],
                            'eval_time_taken': evaluation['time_taken'],
                            'total_cost': bio_data['total_cost'] + evaluation['total_cost'],
                            'total_time_taken': bio_data['time_taken'] + evaluation['time_taken'],
                            'person_biography_length': person_bio_length,
                            'ai_generated_biography_length': ai_bio_length,
                            'generation_attempts': attempt
                        }
                else:
                    log_message(f"AI-generated bio ({ai_bio_length} chars) is shorter than required length ({min_length} chars). Attempting again.")
                    attempt += 1
                    min_length = int(min_length * 1.2)  # Increase the required length by 20% for the next attempt
            else:
                break
        
        log_message(f"Failed to generate a suitable bio for {row.get('FULL_NAME', 'Unknown')} after {max_attempts} attempts.")
        return None
    except KeyError as e:
        log_message(f"Missing data for profile {row.get('FULL_NAME', 'Unknown')}: {str(e)}")
    except Exception as e:
        log_message(f"Error processing profile {row.get('FULL_NAME', 'Unknown')}: {str(e)}")
    return None

def process_csv(csv_file, max_rows=None):
    # Delete existing output file if it exists
    if os.path.exists(excel_filename):
        os.remove(excel_filename)
        log_message(f"Deleted existing output file: {excel_filename}")

    # Check CSV contents
    check_csv_contents(csv_file)

    try:
        # Try reading with comma delimiter
        df = pd.read_csv(csv_file, encoding='utf-8')
        log_message(f"CSV file read successfully with comma delimiter. Shape: {df.shape}")
        log_message(f"Columns: {df.columns.tolist()}")
        log_message(f"First row of data: {df.iloc[0].to_dict()}")
        
        # Check for missing columns
        required_columns = ['PROFILE_ID', 'FULL_NAME', 'LOCATION', 'COMPANY_NAME', 'CURRENT_POSITION', 'PERSON_BIOGRAPHY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            log_message(f"Warning: The following required columns are missing: {missing_columns}")
        
        # Check for empty or all-NA rows
        empty_rows = df[required_columns].isna().all(axis=1)
        if empty_rows.any():
            log_message(f"Warning: {empty_rows.sum()} rows have all NA values in required columns")
        
    except Exception as e:
        log_message(f"Error reading CSV file with comma delimiter: {str(e)}")
        return

    if max_rows:
        df = df.iloc[:max_rows]

    df = df.apply(validate_row, axis=1)

    results = []

    # Use all available CPU cores
    max_workers = multiprocessing.cpu_count()
    log_message(f"Using {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(process_profile, row.to_dict()): row for _, row in df.iterrows()}
        for future in as_completed(future_to_row):
            row_result = future.result()
            if row_result:
                results.append(row_result)

    # Create the output dataframe
    output_df = pd.DataFrame(results)

    # Save to Excel
    with pd.ExcelWriter(excel_filename) as writer:
        output_df.to_excel(writer, sheet_name='Bio Data', index=False)

    log_message(f"\nProcessing complete. Results saved to {excel_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and evaluate professional bios from CSV data")
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
