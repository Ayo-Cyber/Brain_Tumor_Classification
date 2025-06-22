from fastapi import APIRouter, Body, HTTPException
from bson.objectid import ObjectId
from db.db import DatabaseInstantiator
from models.schema import UserIdResponseModel, CreateUser, GetAllUsers, User
from utils.utils import log_message, hash_password

router = APIRouter(prefix="/users")
user_tag = ["users"]
db = DatabaseInstantiator()
user_collection = db.get_table_collection("users")


@router.get("/", tags=user_tag, response_model=GetAllUsers)
async def get_all_users():
    all_users = []
    for user in user_collection.find():
        user["id"] = str(user.pop("_id"))
        all_users.append(UserIdResponseModel(**user))
    return {"users": all_users}


@router.get("/{user_id}", tags=user_tag, response_model=User)
async def get_single_user(user_id: str):
    user_data = user_collection.find_one({"_id": ObjectId(user_id)})
    if not user_data:
        raise HTTPException(status_code=404, detail="user does not exist in the database")
    return user_data


@router.put("/{user_id}", tags=user_tag, response_model=User)
async def update_single_user(user_id: str, data: User = Body(...)):
    user_data = user_collection.find_one_and_update({"_id": ObjectId(user_id)}, {"$set": data.model_dump()})
    return user_data


@router.post("/", tags=user_tag, response_model=UserIdResponseModel)
async def create_user(user: CreateUser = Body(...)):
    user_data = user.model_dump(exclude=["id"])
    check_email = user_collection.find_one({"email": user_data["email"]})
    if check_email:
        raise HTTPException(status_code=404, detail="email already exists")
    user_data["password"] = hash_password(user.password)
    result = user_collection.insert_one(user_data)
    new_user = user_collection.find_one({"_id": result.inserted_id})
    log_message("info", f"successfully executed post request and created new user {new_user['first_name']} {new_user['last_name']}")
    return new_user


@router.delete("/")
async def delete_all_users():
    user_collection.delete_many({})
    return {"status": 200, "detail": "successfully deleted all users"}


@router.delete("/{user_id}", response_model=User)
async def delete_one_user(user_id: str):
    user_to_delete = user_collection.find_one({"_id": ObjectId(user_id)})
    user_collection.delete_one({"_id": ObjectId(user_id)})
    user_to_delete.pop("_id", None)
    user_to_delete.pop("password", None)
    user_to_delete.pop("created_at", None) 
    return User(**user_to_delete)
    
