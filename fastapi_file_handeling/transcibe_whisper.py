from faster_whisper import WhisperModel

def transcibe_audio_from_file(audio_file_path):
    output_text=""

    try:
        model_size="small"
        #loading the model
        model = WhisperModel(model_size,device="cpu",compute_type="int8")
        #transcribing the audio
        segments,info=model.transcribe(audio_file_path,beam_size=5,language="en")
        
        #combining the segments into a single string
        for segment in segments:
            output_text+=segment.text + " "
        
        return output_text,None
    except Exception as e:
        return None,str(e)