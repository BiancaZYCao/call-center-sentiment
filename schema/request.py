""" request data model """
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Define the request model (to handle the incoming JSON request)"""
    type: str
    data: str
    loadingId: str
