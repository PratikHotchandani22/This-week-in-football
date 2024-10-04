from configurations import REDDIT_SOCCER_SUBREDDIT
from scrapping_reddit import initialize_reddit_client, scrape_subreddit, save_scrapped_reddit_data_csvJson
import asyncio

async def main():
    
    reddit = await initialize_reddit_client()
    print("initialized reddit client: ", reddit)
    scrapped_data = await scrape_subreddit(REDDIT_SOCCER_SUBREDDIT,reddit)
    print("scrapped data is: ", scrapped_data)
    print("** Saving scrapped data **")
    save_scrapped_reddit_data_csvJson(scrapped_data)
    await reddit.close()

# Ensure the event loop is run properly
if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function