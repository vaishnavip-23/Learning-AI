# Python has a dynamic type system which means when we create a variable, we don't need to specify the type of the variable.
"""
x=10
x="hello"
"""
# In Python, this is allowed. It takes the value of the last assignment.

"""
def Person(name:str,age:int):
    return {"name":name,"age":age}

print(Person("John",20))
print(Person("John","20")) # different data type but allowed by python

"""
# Using Pydantic : Data Validation Library - IDE Type Hinting, Data Validation, JSON Serialization
# It can take the data directly as function arguments
from pydantic import BaseModel, EmailStr, field_validator, ValidationError
class User(BaseModel):
    name:str
    email:EmailStr
    account_id:int
    # We can use custom validation methods 
    @field_validator("account_id")
    def validate_account_id(cls,value):
        if value <=0:
            raise ValueError(f"Account ID must be positive : {value}")
        return value

try:
    user=User(name="John",email="john@example.com",account_id=10) #validation error
    print(user)
    print(user.model_dump())          # dict
    print(user.model_dump_json())     # JSON string
except ValidationError as e:
    print(e)
"""
user=User(name="John",email="john@example.com",account_id="hello") #validation error
user=User(name="John",email="john@example.com",account_id="25") #automatically converts the string to int
user=User(name="John",email="john@example.com",account_id=25) # works as it should because all types are as specified in the class
"""

"""
user=User(name="John",email="john",account_id=25) # the email is only expecting a string so it passes, but we need to check for email type of we'll use the EmailStr Class. Throws us an error because we're expecting email type now. 

print(user) 
"""


"""
user_data is a dictionary, It also take the data from the dictionary and create the object 
user_data={
    'name':'John',
    'email':'john@example.com',
    'account_id':123
}

user=User(**user_data)
print(user.name)
print(user.email)
print(user.account_id)
"""




    

