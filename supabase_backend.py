from supabase import create_client, Client
from credentials import SUPABASE_KEY, SUPABASE_URL
import asyncio
from itertools import islice

# Initialize the client
async def create_supabase_connection():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase connection created: ", supabase)
    return supabase


# Insert data into the table
async def insert_data_into_table_old(supabase, table_name, job_data_json):
    response = supabase.table(table_name).insert(job_data_json).execute()
    print("Data insertion table name: ")
    print(f"Table name: {table_name}")
    #print(f"Response is: {response}")
    return response

# Function to split the data into batches of a specific size
def chunk_data(data, batch_size):
    """Yield successive batch_size chunks from data."""
    it = iter(data)
    for first in it:
        yield [first] + list(islice(it, batch_size - 1))

# Function to insert data in batches asynchronously
async def insert_data_into_table(supabase, table_name, job_data_json, batch_size=100):
    try:
        # Split the data into batches
        for batch in chunk_data(job_data_json, batch_size):
            # Perform the batch insertion
            response = supabase.table(table_name).insert(batch).execute()
            print(f"Inserted batch into table: {table_name}")
            print(f"Batch size: {len(batch)}")
            # Optional: Handle or log the response
            # print(f"Response: {response}")
            await asyncio.sleep(0.5)  # Small delay to prevent overwhelming the API
        print("All data inserted successfully!")
    except Exception as e:
        print(f"Error during insertion: {e}")
        raise


# Fetch information from the database  
async def fetch_data_from_table(supabase, table_name):
    print(f"Fetching data from table: {table_name}")
    response = supabase.table('job_info').select('*').execute()
    print(f"Response is: {response}")
    return response



