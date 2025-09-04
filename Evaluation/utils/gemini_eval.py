import os
from io import BytesIO
import vertexai
from PIL import Image as PILImage
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel, Part

# Your configs
generation_config = {
    "max_output_tokens": 2048,
    "temperature": 0.4,
    "top_p": 0.4,
    "top_k": 32,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH:
        generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
        generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
        generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT:
        generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './gemini_key.json'
# 1) Init Vertex AI
vertexai.init(project="tencent-gemini-omd01", location="us-central1")

# 2) Load the multimodal model
# model = GenerativeModel("gemini-2.5-pro-preview-03-25")
model = GenerativeModel("gemini-2.0-flash")
# model = GenerativeModel("gemini-2.0-flash-lite")


prompt_template = '''You are provided a question, a gold answer, and a candidate answer. Your task is to judge the correctness of the candidate answer. Return your judgment enclosed with <judgment> </judgment>.\nQuestion:{Question}\nReference Answer: {Reference}\nCandidate Answer: {Candidate}'''

def generate(prompt_question, reference, candidate):
    prompt_question = prompt_question.replace('<image>', '')
    # reference = extract_answer(reference)
    reference = reference
    prompt_message = prompt_template.replace('{Question}', prompt_question).replace('{Reference}', reference).replace('{Candidate}', candidate)
    try:
        # 5) Generate with image + text prompt
        responses = model.generate_content(
                    contents=[
                    prompt_message
                    ],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True,
        )

        # 6) Print streamed output
        full = ""
        for chunk in responses:
            full += chunk.text
                
        return full
    except Exception as e:
        print(e)
        return "error"
                
                