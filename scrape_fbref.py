# league_scraper.py
from configurations import leagues
import pandas as pd
import re
import asyncio
import aiohttp
import nest_asyncio

# Apply nest_asyncio to allow nested event loops in environments like Jupyter
nest_asyncio.apply()


async def fetch_and_clean_league_data(session, league_name, url, table_id):
    async with session.get(url) as response:
        html = await response.text()
        df = pd.read_html(html, attrs=table_id)[0]
        
        # Drop rows with NaN in 'Wk' column
        df = df.dropna(subset=['Wk'])
        
        # Remove words starting with lowercase letters in 'Home' and 'Away' columns
        df['Home'] = df['Home'].apply(lambda x: ' '.join([word for word in x.split() if not word[0].islower()]))
        df['Away'] = df['Away'].apply(lambda x: ' '.join([word for word in x.split() if not word[0].islower()]))
        df.rename(columns={'xG': 'xG_home', 'xG.1':"xG_away"}, inplace=True)
        # Combine date and time into a single timestamp with time zone
        df['match_date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        
        # Add the league name as a new column
        df['league_name'] = league_name
        
        return df

async def scrape_and_process_leagues():
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Iterate over each league and add a 2-second delay between requests
        for league_name, info in leagues.items():
            task = fetch_and_clean_league_data(session, league_name, info['url'], info['table_id'])
            tasks.append(task)
            await asyncio.sleep(2)  # 2-second delay
        
        # Gather all results once scraping is complete
        dfs = await asyncio.gather(*tasks)
        
        # Concatenate all dataframes
        final_df = pd.concat(dfs, ignore_index=True)

        # 1. Drop the 'Notes' and 'Match Report' columns if they exist
        for column in ['Notes', 'Match Report']:
            if column in final_df.columns:
                final_df = final_df.drop(columns=[column])

        # 2. Drop rows where the 'Score' column is empty (NaN or empty string)
        final_df = final_df.dropna(subset=['Score']).loc[final_df['Score'] != '']

        # 3. Create a unique identifier using 'Date', 'Home', and 'Away' columns
        final_df['unique_identifier_id'] = final_df['Date'].astype(str) + "_" + final_df['Home'] + "_" + final_df['Away']

        final_df.columns = final_df.columns.str.lower()

        # Drop the 'Date' and 'Time' columns
        final_df = final_df.drop(columns=['date', 'time'])

        # Drop the last row
        final_df = final_df.drop(final_df.index[-1])  # Drop the last row using its index

        # Save to CSV
        final_df.to_csv("merged_league_data.csv", index=False)
        print("Data scraped, cleaned, and saved as 'merged_league_data.csv'")

        return final_df