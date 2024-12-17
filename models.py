from pydantic import BaseModel

class AnswerFormat(BaseModel):
    final_answer: int

class ConfidenceFormat(BaseModel):
    confidence: float
