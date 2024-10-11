from supabase import create_client, Client
from credentials import SUPABASE_KEY, SUPABASE_URL


# Initialize the client
async def create_supabase_connection():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase connection created: ", supabase)
    return supabase


# Insert data into the table
def insert_data_into_table(supabase, table_name, job_data_json):
    response = supabase.table(table_name).insert(job_data_json).execute()
    print("Data insertion table name: ")
    print(f"Table name: {table_name}")
    #print(f"Response is: {response}")
    return response

# Fetch information from the database  
def fetch_data_from_table(supabase, table_name):
    print(f"Fetching data from table: {table_name}")
    response = supabase.table('job_info').select('*').execute()
    print(f"Response is: {response}")
    return response



