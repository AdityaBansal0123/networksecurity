from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

url = "mongodb+srv://bansaladitya048:Aditya123@cluster0.i9jzs.mongodb.net/?appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(url, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)