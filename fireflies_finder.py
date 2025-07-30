# This script contains functions to interact with the Fireflies API, handling transcript searches, data formatting, and rate-limiting.
import os
import requests
import json
from datetime import datetime, timedelta
import logging
import sys
from typing import List, Tuple, Dict, Set, Optional
import re
from dotenv import load_dotenv
from urllib.parse import urlparse
from collections import deque
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fireflies_finder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Fireflies API configuration
FIREFLIES_API_URL = 'https://api.fireflies.ai/graphql'
FIREFLIES_TOKEN = os.getenv("FIREFLIES_TOKEN")
if not FIREFLIES_TOKEN:
    raise ValueError("FIREFLIES_TOKEN environment variable is required. Please set it in your .env file.")

class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()  # Store timestamps of recent requests

    def can_make_request(self):
        """Check if we can make a request without hitting rate limit"""
        now = datetime.now()

        # Remove requests older than 1 minute
        while self.request_times and (now - self.request_times[0]) > timedelta(minutes=1):
            self.request_times.popleft()

        # Check if we're under the limit
        return len(self.request_times) < self.max_requests

    def wait_if_needed(self):
        """Wait if we're approaching the rate limit"""
        while not self.can_make_request():
            # Wait until the oldest request is more than a minute old
            wait_time = (self.request_times[0] + timedelta(minutes=1) - datetime.now()).total_seconds()
            if wait_time > 0:
                print(f"Rate limit approaching. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            self.can_make_request() # re-check after waiting

        # Record this request
        self.request_times.append(datetime.now())

# Initialize global rate limiter
rate_limiter = RateLimiter(max_requests_per_minute=50) # A bit under 60 to be safe

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase, trimming whitespace,
    and removing common punctuation.
    """
    if not text:
        return ""
    # Convert to lowercase and trim
    text = text.lower().strip()
    # Remove common punctuation
    text = re.sub(r'[,\.&]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text

def make_graphql_request(query: str, variables: Optional[Dict] = None) -> Optional[Dict]:
    """
    Make a GraphQL request to the Fireflies API with rate limiting.
    """
    rate_limiter.wait_if_needed()
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {FIREFLIES_TOKEN}'
    }
    
    payload = {
        'query': query,
        'variables': variables or {}
    }
    
    try:
        response = requests.post(FIREFLIES_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        if 'errors' in result:
            logger.error("GraphQL Errors:")
            for error in result['errors']:
                logger.error(f"- {error.get('message')}")
            return None
            
        return result
    except Exception as e:
        logger.error(f"Error making GraphQL request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                logger.error(f"API Error details: {json.dumps(error_data, indent=2)}")
            except json.JSONDecodeError:
                logger.error(f"Response text: {e.response.text}")
        return None

def search_fireflies(query: str, is_email_search: bool = False) -> Tuple[List[Dict], Optional[str]]:
    """
    Search Fireflies API using GraphQL for meetings matching the query.
    If is_email_search is True, searches in participants' emails using the participant_email argument.
    Returns a tuple of (transcripts, next_cursor).
    """
    if is_email_search:
        search_query = '''
        query SearchTranscriptsByParticipant($email: String!) {
            transcripts(participant_email: $email) {
                id
                title
                date
                duration
                host_email
                organizer_email
                transcript_url
                participants
                speakers {
                    id
                    name
                }
                sentences {
                    speaker_name
                    text
                    start_time
                    end_time
                    raw_text
                }
            }
        }
        '''
        variables = { 'email': query }
    else:
        search_query = '''
        query SearchTranscripts($searchTerm: String!) {
            transcripts(title: $searchTerm) {
                id
                title
                date
                duration
                host_email
                organizer_email
                transcript_url
                participants
                speakers {
                    id
                    name
                }
                sentences {
                    speaker_name
                    text
                    start_time
                    end_time
                    raw_text
                }
            }
        }
        '''
        variables = {
            'searchTerm': query
        }
    
    result = make_graphql_request(search_query, variables)
    if not result or 'data' not in result:
        logger.error("No data in response")
        return [], None
        
    if 'transcripts' not in result['data']:
        logger.error("No transcripts field in response data")
        return [], None
    
    transcripts = result['data']['transcripts']
    if not isinstance(transcripts, list):
        transcripts = [transcripts]
    
    return transcripts, None

def fetch_all_transcripts(search_func, query: str, is_email_search: bool = False) -> List[Dict]:
    """
    Fetch transcripts using the search function.
    """
    transcripts, _ = search_func(query, is_email_search)
    return transcripts

def format_meeting_data(meeting: Dict) -> str:
    """
    Format meeting data into a string for file output.
    """
    try:
        date = datetime.fromtimestamp(meeting['date'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        duration = round(meeting['duration'], 2)
        
        output = f"""Title: {meeting['title']}
Date: {date}
Duration: {duration} minutes
Transcript URL: {meeting['transcript_url']}
Host: {meeting.get('host_email', 'Not specified')}
Organizer: {meeting.get('organizer_email', 'Not specified')}
{'='*50}

"""
        # Add speakers if available
        speakers = meeting.get('speakers', [])
        if speakers:
            output += "Speakers:\n"
            for speaker in speakers:
                output += f"- {speaker.get('name')} (ID: {speaker.get('id')})\n"
            output += "\n"
        
        # Add transcript content without timestamps
        sentences = meeting.get('sentences', [])
        if sentences:
            output += "Transcript Content:\n"
            output += "-" * 50 + "\n"
            for sentence in sentences:
                speaker = sentence.get('speaker_name', 'Unknown')
                text = sentence.get('text', '')
                output += f"{speaker}: {text}\n"
            output += "\n" + "=" * 50 + "\n\n"
        else:
            output += "No transcript content available\n"
            output += "\n" + "=" * 50 + "\n\n"
        
        return output
    except Exception as e:
        logger.error(f"Error formatting meeting data: {str(e)}")
        return f"""Title: {meeting.get('title', 'Unknown')}
Date: Unknown
Duration: Unknown
Transcript URL: Unknown
{'='*50}
"""

def is_company_in_transcript(meeting: Dict, company_name: str) -> bool:
    """
    Check if company name appears in the meeting transcript (case-insensitive).
    """
    company_name = company_name.lower()
    # Check title
    if company_name in meeting.get('title', '').lower():
        return True
    # Check transcript content
    for sentence in meeting.get('sentences', []):
        if company_name in sentence.get('text', '').lower():
            return True
    return False

def has_email_participant(participants: List[str], email: str) -> bool:
    """
    Check if the exact email address is in the participants list.
    """
    email = email.lower()
    return email in [participant.lower() for participant in participants if '@' in participant]

def process_company(company_name: str, emails: List[str]) -> None:
    """
    Process a single company: search Fireflies and save results to file.
    """
    # Get unique meeting IDs to avoid duplicates
    unique_meetings: Dict[str, Dict] = {}  # Using dict for better lookup performance
    duplicate_count = 0
    total_email_matches = 0  # Total matches by email, including duplicates
    
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
        
        # Search by company name first
        print(f"\nSearching for company name '{company_name}' in transcripts...")
        name_results = fetch_all_transcripts(search_fireflies, company_name, is_email_search=False)
        print(f"Retrieved {len(name_results)} potential matches by company name")
        
        # Add all meetings where company name appears in transcript
        name_matches = 0
        for meeting in name_results:
            if is_company_in_transcript(meeting, company_name):
                unique_meetings[meeting['id']] = {
                    'meeting': meeting,
                    'found_by_name': True,
                    'found_by_email': False,
                    'matching_emails': set()  # Track which emails matched
                }
                name_matches += 1
                print(f"✓ Found matching transcript by company name: {meeting['title']}")
        
        print(f"Confirmed {name_matches} transcripts contain company name")
        
        # Search by each email address
        new_email_matches = 0  # Matches found only by email
        for email in valid_emails:
            print(f"\nSearching for email '{email}' in participant emails...")
            email_results = fetch_all_transcripts(search_fireflies, email, is_email_search=True)
            print(f"Retrieved {len(email_results)} potential matches for email")
            
            email_matches = 0
            # Add all meetings where email appears in participants
            for meeting in email_results:
                if has_email_participant(meeting.get('participants', []), email):
                    if meeting['id'] in unique_meetings:
                        unique_meetings[meeting['id']]['found_by_email'] = True
                        unique_meetings[meeting['id']]['matching_emails'].add(email)
                        duplicate_count += 1
                        print(f"↺ Duplicate found: {meeting['title']}")
                        print(f"  - Already found by company name, also matches email: {email}")
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
        print(f"- Matches by company name: {name_matches}")
        print(f"- Matches by email: {total_email_matches}")
        print(f"  └─ New matches (email only): {new_email_matches}")
        print(f"  └─ Duplicate matches (also matched company): {duplicate_count}")
        print(f"- Total unique transcripts: {len(unique_meetings)}")
        
        # Count transcripts with content
        with_content = sum(1 for m in unique_meetings.values() if m['meeting'].get('sentences'))
        print(f"- Transcripts with content: {with_content}")
        
        # Create output directory if it doesn't exist
        os.makedirs('transcripts', exist_ok=True)
        
        # Save results to file
        filename = f"transcripts/{company_name}_all_transcripts.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Transcripts for {company_name}\n")
            f.write(f"Contact Emails: {', '.join(valid_emails)}\n")
            f.write("=" * 50 + "\n\n")
            
            # Write all meetings with their match type
            for meeting_data in unique_meetings.values():
                meeting = meeting_data['meeting']
                match_type = []
                if meeting_data['found_by_name']:
                    match_type.append("Company Name in Transcript")
                if meeting_data['found_by_email']:
                    emails = ', '.join(meeting_data['matching_emails'])
                    match_type.append(f"Email in Participants ({emails})")
                
                f.write(f"Match Type: {' & '.join(match_type)}\n")
                f.write(format_meeting_data(meeting))
        
        print(f"\nSaved all transcripts to: {filename}")
        
    except Exception as e: 
        logger.error(f"Error processing company {company_name}: {str(e)}")
        raise

def main():
    """
    Main function to process companies with manual input.
    """
    print("\nEnter company information (press Enter with empty company name to finish):")
    manual_companies = []
    
    while True:
        company_name = input("\nEnter company name (or press Enter to finish): ").strip()
        if not company_name:
            break
            
        emails = input("Enter contact emails (comma-separated): ").strip()
        manual_companies.append({
            "Company Name": company_name,
            "Contact Emails": emails
        })
    
    if not manual_companies: 
        return
    
    # Process each company
    for company in manual_companies:
        company_name = company["Company Name"]
        contact_emails = [email.strip() for email in company["Contact Emails"].split(",") if email.strip()]
        process_company(company_name, contact_emails)

if __name__ == "__main__":
    main() 
































