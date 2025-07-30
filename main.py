import sys
#main: connects hubspot_fireflies_integration.py and keyword_pull.py + fireflies_finder.py as a pipeline
# Step 1: Run the HubSpot + Fireflies integration to fetch transcripts
from processor_final.hubspot_fireflies_integration import main as hubspot_fireflies_main
# Step 2: Run the keyword miner on the transcripts
from processor_final.keyword_pull import main as keyword_pull_main


def main():
    print("Step 1: Running HubSpot + Fireflies integration...")
    hubspot_fireflies_main()
    print("\nStep 2: Running keyword miner on transcripts...")
    keyword_pull_main()
    print("\nAll done!")

if __name__ == "__main__":
    main()





















