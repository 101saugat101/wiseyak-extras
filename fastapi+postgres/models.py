from sqlalchemy import Column, Integer, String, Text, DateTime
from database import Base

class student(Base):
    __tablename__ = "student"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    age = Column(Integer, nullable=False)
    