# transcript_processor_final

HubSpot + Fireflies Transcript Miner - README


##Overview:
This is a Python-based pipeline that connects HubSpot CRM with Fireflies.ai to:

Search company names and contact emails from HubSpot.

Retrieve meeting transcripts from Fireflies.ai.

Extract important keywords using TF-IDF (with spaCy and scikit-learn).

Output cleaned transcript files and keyword tables.


##Environment Setup:

Install required packages:
pip install -U spacy requests prettytable numpy scikit-learn python-dotenv pandas
python -m spacy download en_core_web_sm

Create a .env file and add your API tokens:
HUBSPOT_TOKEN=your_hubspot_api_token
FIREFLIES_TOKEN=your_fireflies_api_token


##Folder Structure:

transcripts/: contains all output transcript files per company

keyword_universe.csv and .xlsx: files with extracted keyword stats

processor_final/: source code folder


##Files and Roles:

fireflies_finder.py: Handles Fireflies API transcript searches by company name and email.

hubspot_fireflies_integration.py: Queries HubSpot for companies and retrieves associated contact emails.

keyword_pull.py: Runs TF-IDF keyword mining on transcript content.

main.py: Orchestrates the entire pipeline (HubSpot -> Fireflies -> Keyword Mining).


##How to Use:

Set your API tokens in a .env file.

Run the main script:
python main.py

Follow prompts to enter company names (one per line).

Transcripts will be saved in the transcripts/ folder.

Keywords will be extracted and saved to keyword_universe.csv and keyword_universe.xlsx.


##Output:

For each company, a transcript file is created with metadata, participants, and full transcript text.

The keyword miner outputs top uni- to quad-gram phrases ranked by TF-IDF.


##Important Notes:

Transcript duplication is automatically avoided.

API rate limits are limited based on fireflies subscription (max 50 requests/min)


