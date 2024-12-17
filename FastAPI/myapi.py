from fastapi import FastAPI, Path, HTTPException
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

students = {
    1: {"name": "saugat", "age": 17, "class": "year 12"}
}

class BaseStudent(BaseModel):
    name: str
    age: int            
    year:str


class UpdateStudent(BaseModel):
    name: Optional[str]=None
    age: Optional[int]=None   
    year: Optional[str]=None

@app.get("/")
def index():
    return {"name": "first name"}

@app.get("/get-student/{student_id}")
def get_student(student_id: int = Path(..., description="Student ID hereeeeeee",gt=0)):
    return students[student_id]


@app.get("/get-by-name/{student_id}")
def get_by_name(*,student_id:int,name: Optional[str] = None, test: int):
    for student_id, student in students.items():
        if student["name"] == name:
            return student
    raise HTTPException(status_code=404, detail="Student not found")


@app.post("/add-student/{student_id}")
def add_student(*,student_id: int = Path(..., description="Student ID hereeeeeee",gt=0), student: BaseStudent):
    if student_id in students:
        return {"message": "Student already exists"}
    students[student_id] = student.dict()
    return {"message": "Student added successfully"}


@app.put("/update-student/{student_id}")
def update_student(*,student_id: int = Path(..., description="Student ID hereeeeeee",gt=0), student: UpdateStudent):
    if student_id not in students:
        return {"message": "Student not found"}

    if student.name!=None:
        students[student_id]["name"]=student.name

    if student.age!=None:
        students[student_id]["age"]=student.age
    
    if student.year!=None:
        students[student_id]["year"]=student.year

    
    return students[student_id]


@app.delete("/delete-student/{student_id}")
def delete_student(*,student_id: int = Path(..., description="Student ID hereeeeeee",gt=0)):
    if student_id not in students:
        return {"message": "Student not found"}
    del students[student_id]
    return {"message": "Student deleted successfully"}