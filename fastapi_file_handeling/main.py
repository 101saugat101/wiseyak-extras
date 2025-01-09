import os 
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException,BackgroundTasks
from pydantic import BaseModel
from typing import List
from starlette.responses import JSONResponse
from transcibe_whisper import transcibe_audio_from_file
from llm_summarize import summarize_text

#initializing the app
app = FastAPI()

#list to store the audio files transciption
transciption_tasks=[]


#defining the model for the audio transciption
class TranscriptionModelResult(BaseModel):
    task_id: str   
    transciption: str=None
    summary: str=None
    tags: List[str]=[]
    discription: str=None
    status: str="processing"
    message: str=None
    error: str=None


#background task to process the audio transciption and summrization

def background_task(file_location:str,task_id:str):
    #creating a new transciption task
    new_task=TranscriptionModelResult(task_id=task_id)
    #adding the task to the list of transciption tasks
    transciption_tasks.append(new_task)
    #transcibing the audio file
    transciption,error=transcibe_audio_from_file(file_location)
    if error:
        new_task.status="error"
        new_task.message=error
        return
    else:
        """#i will update this section later"""

    #updating the task status
    new_task.status="completed"
    os.remove(file_location)


#endpoint to start process the audio file transciption
@app.post("/process_audio")
async def process_audio(background_tasks:BackgroundTasks,file:UploadFile=File(...)):
    #saving file to disk
    file_location=f"./audio/{file.filename}"
    with open(file_location,"wb") as f:
        f.write(file.file.read())
    
    #generate a unique task id
    task_id = str(uuid.uuid4())
    #adding the task to the background tasks
    background_tasks.add_task(background_task,file_location,task_id)
    #returning the task id
    return JSONResponse(content={"task_id":task_id},status_code=201)

#endpoint to return status and result of the transciption task
@app.get("/transciption_task/{task_id}",response_model=TranscriptionModelResult)
async def get_transcription_task(task_id:str):
    #searching for the task in the list of transciption tasks and returning the result
    task=next((task for task in transciption_tasks if task.task_id==task_id),None)
    if task is None:
        raise HTTPException(status_code=404,detail="Task not found")
    return JSONResponse(content=task.dict(),status_code=200)


#running fastapi app
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)

