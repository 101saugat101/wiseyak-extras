import os
import json
from llama_cpp import Llama

#setuping the lama and mistral
llm = Llama(model_path="models/mistral-7b-instruct.Q4_K_M.ggmlv3.q4_0.bin",chat_format="llama-2",cpu_threads=os.cpu_count(),n_ctx=4096)

def summarize_text(transciption):
    try:
        result=llm.create_chat_completion(
            messages=[
                {"role": "system", "content":"""you are assistant that recieves transciption and generates summary
                please generate summary in format
                {
                "summary":"summary of transciption",
                "discription":"description of transciption"
                }"""},
                {"role": "user", "content":transciption}
            ])

        return json.loads(result)
    except Exception as e:
        return None,str(e)







