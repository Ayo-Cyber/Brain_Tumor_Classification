from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import hashlib
import streamlit as st

# Connect to MongoDB Atlas
client = MongoClient('mongodb+srv://atunraseayomide:mcBi5DCZ0JBRQUvy@braintumorclassifier.mqezn0m.mongodb.net/')
db = client['Brain_Tumor_Project']  
users_collection = db['users']
images_collection = db['images']

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    hashed_password = hash_password(password)
    user_data = {
        'username': username,
        'password': hashed_password,
        'email': email
    }
    try:
        users_collection.insert_one(user_data)
        return True
    except DuplicateKeyError:
        return False

def login_user(username, password):
    hashed_password = hash_password(password)
    user = users_collection.find_one({'username': username, 'password': hashed_password})
    return user

def save_image(user_id, image_path, predicted_class, confidence):
    image_data = {
        'user_id': user_id,
        'image_path': image_path,
        'predicted_class': predicted_class,
        'confidence': confidence
    }
    images_collection.insert_one(image_data)

def get_user_images(user_id):
    return list(images_collection.find({'user_id': user_id}))

# Test MongoDB Connection
def test_mongo_connection():
    try:
        # Try a simple query to the 'users' collection
        db.command('ping')  # MongoDB 4.0+ has the "ping" command to check connectivity
        st.success("MongoDB connection is successful!")
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")

# Test Database and Collections
def test_db_and_collections():
    try:
        # Test if database and collection exist
        if 'Brain_Tumor_Project' in client.list_database_names():
            st.success("Database 'Brain_Tumor_Project' found!")
            if 'users' in db.list_collection_names():
                st.success("Collection 'users' found!")
            else:
                st.error("Collection 'users' not found.")
            if 'images' in db.list_collection_names():
                st.success("Collection 'images' found!")
            else:
                st.error("Collection 'images' not found.")
        else:
            st.error("Database 'Brain_Tumor_Project' not found.")
    except Exception as e:
        st.error(f"Error checking database or collections: {e}")

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state['user_id'] = str(user['_id'])
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password")

def register_page():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")
    if st.button("Register"):
        if register_user(username, password, email):
            st.success("Registered successfully! Please login.")
        else:
            st.error("Username or email already exists")

def main():
    # Test MongoDB connection and collections
    test_mongo_connection()
    test_db_and_collections()

    if 'user_id' not in st.session_state:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Login", "Register"])
        if page == "Login":
            login_page()
        elif page == "Register":
            register_page()
    else:
        st.write(f"Welcome, User ID: {st.session_state['user_id']}")
        # Add your main application logic here

if __name__ == "__main__":
    main()
