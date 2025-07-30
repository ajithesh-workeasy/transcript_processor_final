# This script integrates HubSpot and Fireflies by searching for companies in HubSpot, finding associated contact emails, and then fetching corresponding transcripts from Fireflies.
import os
import sys
import logging
import time
from typing import List
from dotenv import load_dotenv
import argparse
from collections import deque
from datetime import datetime, timedelta

# Import HubSpot and Fireflies logic
import requests
from datetime import datetime

# Import process_company from fireflies_finder
from processor_final.fireflies_finder import process_company


# Conservative API usage configuration
CONSERVATIVE_CONFIG = {
    'max_emails_per_company': 10,        # Limit emails searched per company
    'sufficient_results_threshold': 999, # Always do email searches (set to high number)
    'skip_companies_without_emails': True,  # Skip companies with no contact emails
    'max_companies_per_batch': 20,      # Process companies in smaller batches
}


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hubspot_fireflies.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# HubSpot API token
HUBSPOT_TOKEN = os.getenv("HUBSPOT_TOKEN")
if not HUBSPOT_TOKEN:
    raise ValueError("HUBSPOT_TOKEN environment variable is required. Please set it in your .env file.")

def search_company_by_name(company_name: str):
    """
    Search for a specific company by name and get its details and associated contact emails
    """
    url = "https://api.hubapi.com/crm/v3/objects/companies/search"
    headers = {
        "Authorization": f"Bearer {HUBSPOT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "filterGroups": [
            {
                "filters": [
                    {
                        "propertyName": "name",
                        "operator": "CONTAINS_TOKEN",
                        "value": company_name
                    }
                ]
            }
        ],
        "limit": 10,
        "properties": ["name", "domain", "recent_deal_close_date"]
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        companies = response.json().get("results", [])
        if not companies:
            logger.warning(f"No companies found matching '{company_name}'")
            return None
        logger.info(f"Found {len(companies)} companies matching '{company_name}'")
        company = companies[0]
        company_id = company["id"]
        official_name = company["properties"].get("name", "N/A")
        domain = company["properties"].get("domain", "N/A")
        close_date = company["properties"].get("recent_deal_close_date", "N/A")
        logger.info(f"Processing company: {official_name}")
        contacts_url = f"https://api.hubapi.com/crm/v3/objects/companies/{company_id}/associations/contacts"
        contacts_response = requests.get(contacts_url, headers=headers)
        contacts_response.raise_for_status()
        contact_emails = []
        for contact in contacts_response.json().get("results", []):
            contact_id = contact["id"]
            contact_url = f"https://api.hubapi.com/crm/v3/objects/contacts/{contact_id}"
            contact_response = requests.get(contact_url, headers=headers)
            contact_response.raise_for_status()
            contact_data = contact_response.json()
            email = contact_data["properties"].get("email", "N/A")
            if email != "N/A":
                contact_emails.append(email)
        return {
            "Official Name": official_name,
            "Domain": domain,
            "Close Date": close_date,
            "Contact Emails": contact_emails
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {str(e)}")
        return None

def main():
    print("HubSpot and Fireflies Transcript Search")
    print("Enter up to 100 company names (one per line) in a single input:")
    print("Example:")
    print("Apple Inc")
    print("Microsoft Corporation")
    print("Google LLC")
    print("(Press Enter twice when done)")
    
    company_names = []
    print("\nEnter your company names:")
    
    # Read all input at once
    while len(company_names) < 100:
        try:
            line = input().strip()
            if line == "":
                break
            if line.lower() in ['done', 'exit', 'quit']:
                break
            if line:
                company_names.append(line)
        except EOFError:
            break
    
    if not company_names:
        print("No companies entered. Exiting.")
        return
    
    print(f"\nProcessing {len(company_names)} companies...")
    
    # Create output directory for all transcript files
    output_dir = "transcripts"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All transcript files will be saved to: {output_dir}/")
    
    # Get list of existing files to avoid duplicates
    existing_files = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('_all_transcripts.txt'):
                existing_files.add(filename)
    
    print(f"Found {len(existing_files)} existing transcript files in {output_dir}/")
    
    summary = []
    for idx, company_name in enumerate(company_names, 1):
        logger.info(f"[{idx}/{len(company_names)}] Searching for company: {company_name}")
        print(f"\n[{idx}/{len(company_names)}] Searching for company: {company_name}")
        company_data = search_company_by_name(company_name)
        if not company_data:
            print(f"No company found matching '{company_name}' or an error occurred.")
            summary.append((company_name, 'Not found or error', 0))
            continue
        print(f"Company found: {company_data['Official Name']}")
        
        # Clean and validate emails, removing trailing punctuation
        cleaned_emails = [
            email.strip().rstrip('.,') 
            for email in company_data['Contact Emails'] 
            if isinstance(email, str) and email
        ]
        valid_emails = [email for email in cleaned_emails if '@' in email]
        
        # Exclude workeasysoftware.com emails from being searched
        valid_emails = [
            email for email in valid_emails
            if not email.lower().endswith('@workeasysoftware.com')
        ]
        
        # Limit emails to reduce API calls (take first 3 most relevant emails)
        max_emails_per_company = CONSERVATIVE_CONFIG['max_emails_per_company']
        if len(valid_emails) > max_emails_per_company:
            print(f"Limiting to first {max_emails_per_company} emails to reduce API calls")
            valid_emails = valid_emails[:max_emails_per_company]
        
        # Skip companies without emails if configured
        if not valid_emails and CONSERVATIVE_CONFIG['skip_companies_without_emails']:
            print(f"No valid emails found for '{company_name}'. Skipping to save API calls.")
            continue
        
        # Check if transcript file already exists
        expected_filename = f"{company_data['Official Name']}_all_transcripts.txt"
        if expected_filename in existing_files:
            print(f"Transcript file already exists for '{company_data['Official Name']}'. Skipping to avoid duplicates.")
            summary.append((company_data['Official Name'], 'Already exists', len(valid_emails)))
            continue
        
        print(f"Number of contacts: {len(valid_emails)}")
        print("Searching Fireflies for transcripts using company and contacts...")
        try:
            # Pass the output directory to process_company
            process_company_with_output_dir(company_data['Official Name'], valid_emails, output_dir)
            print("Done. Check the transcripts folder for results.")
            summary.append((company_data['Official Name'], 'Processed', len(valid_emails)))
        except Exception as e:
            logger.error(f"Error processing Fireflies for {company_data['Official Name']}: {str(e)}")
            print(f"Error processing Fireflies for {company_data['Official Name']}: {str(e)}")
            summary.append((company_data['Official Name'], 'Fireflies error', len(valid_emails)))

    print("\nSummary:")
    for name, status, num_contacts in summary:
        print(f"{name}: {status} (Contacts: {num_contacts})")

def process_company_with_output_dir(company_name: str, emails: List[str], output_dir: str) -> None:
    """
    Process a single company: search Fireflies and save results to specified directory.
    Modified version of process_company from fireflies_finder.py
    """
    # Import the necessary functions from fireflies_finder
    from processor_final.fireflies_finder import fetch_all_transcripts, search_fireflies, is_company_in_transcript, has_email_participant, format_meeting_data
    
    # Get unique meeting IDs to avoid duplicates
    unique_meetings: Dict[str, Dict] = {}
    duplicate_count = 0
    total_email_matches = 0
    
    try:
        print(f"\nProcessing company: {company_name}")
        
        # Clean and validate emails
        valid_emails = [email.strip() for email in emails if '@' in email.strip()]
        
        # Exclude workeasysoftware.com emails from being searched
        valid_emails = [
            email for email in valid_emails
            if not email.lower().endswith('@workeasysoftware.com')
        ]
        
        print(f"Found valid emails: {', '.join(valid_emails)}")
        
        # Skip companies without emails if configured
        if not valid_emails and CONSERVATIVE_CONFIG['skip_companies_without_emails']:
            print(f"No valid emails found for '{company_name}'. Skipping to save API calls.")
            return
        
        # Search by each email address only (skip company name search)
        new_email_matches = 0
        for email in valid_emails:
            print(f"\nSearching for email '{email}' in participant emails...")
            email_results = fetch_all_transcripts(search_fireflies, email, is_email_search=True)
            print(f"Retrieved {len(email_results)} potential matches for email")
            
            email_matches = 0
            for meeting in email_results:
                if has_email_participant(meeting.get('participants', []), email):
                    if meeting['id'] in unique_meetings:
                        unique_meetings[meeting['id']]['found_by_email'] = True
                        unique_meetings[meeting['id']]['matching_emails'].add(email)
                        duplicate_count += 1
                        print(f"↺ Duplicate found: {meeting['title']}")
                        print(f"  - Already found by another email: {email}")
                    else:
                        unique_meetings[meeting['id']] = {
                            'meeting': meeting,
                            'found_by_name': False,
                            'found_by_email': True,
                            'matching_emails': {email}
                        }
                        new_email_matches += 1
                        print(f"✓ Found new matching transcript by email: {meeting['title']}")
                    email_matches += 1
                    total_email_matches += 1
            
            print(f"Confirmed {email_matches} transcripts contain email {email}")
        
        if not unique_meetings:
            print(f"\nNo matching transcripts found for {company_name}")
            return
        
        # Summary statistics
        print(f"\nSearch Summary for {company_name}:")
        print(f"- Matches by email: {total_email_matches}")
        print(f"- Total unique transcripts: {len(unique_meetings)}")
        
        # Count transcripts with content
        with_content = sum(1 for m in unique_meetings.values() if m['meeting'].get('sentences'))
        print(f"- Transcripts with content: {with_content}")
        
        # Save results to file in the specified output directory
        filename = os.path.join(output_dir, f"{company_name}_all_transcripts.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Transcripts for {company_name}\n")
            f.write(f"Contact Emails: {', '.join(valid_emails)}\n")
            f.write("=" * 50 + "\n\n")
            
            # Write all meetings with their match type
            for meeting_data in unique_meetings.values():
                meeting = meeting_data['meeting']
                match_type = []
                if meeting_data['found_by_email']:
                    emails = ', '.join(meeting_data['matching_emails'])
                    match_type.append(f"Email in Participants ({emails})")
                
                f.write(f"Match Type: {' & '.join(match_type)}\n")
                f.write(format_meeting_data(meeting))
        
        print(f"\nSaved all transcripts to: {filename}")
        
    except Exception as e:
        logger.error(f"Error processing company {company_name}: {str(e)}")
        raise

if __name__ == "__main__":
    main()


    

 



    


