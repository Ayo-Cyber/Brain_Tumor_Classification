from pydantic import BaseModel, Field, EmailStr, SecretStr, ConfigDict
from datetime import datetime
from typing_extensions import Annotated
from pydantic.functional_validators import BeforeValidator
from typing import List
PyObjectId = Annotated[str, BeforeValidator(str)]


class User(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    email: EmailStr
    first_name: str = Field(min_length=3, max_length=10)
    last_name: str = Field(min_length=3, max_length=10)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "username": "janedoe",
                "email": "jane@gmail.com",
                "first_name": "Jane",
                "last_name": "Doe",
                "password": "tst"
            }
        },
    )


class CreateUser(User):
    password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Predictions(BaseModel):
    image_url: str
    prediction: str
    prediction_time: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "1",
                "image_url": "s3://url.com",
                "prediction": "glioma",
                "prediction_time": "2025-03-05T12:00:00",
            }
        }
    )


class UserResponseModel(BaseModel):
    status: str
    user: User


class UserIdResponseModel(User):
    id: PyObjectId = Field(alias="_id")


class GetAllUsers(BaseModel):
    users: List[UserIdResponseModel]
