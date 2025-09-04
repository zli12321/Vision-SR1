import os
from openai import AzureOpenAI
import time
import base64
from mimetypes import guess_type
import re
from typing import Dict, List, Optional

def extract_answer(predict: str) -> Optional[str]:
    """
    Extracts the content of the <answer>…</answer> block from `predict`.
    Returns the inner text (with leading/trailing whitespace stripped),
    or None if no <answer> tag is found.
    """
    match = re.search(r"<answer>([\s\S]*?)</answer>", predict, re.DOTALL)
    if not match:
        return predict
    return match.group(1).strip()

def extract_judgment(predict: str) -> Optional[str]:
    """
    Extracts the content of the <answer>…</answer> block from `predict`.
    Returns the inner text (with leading/trailing whitespace stripped),
    or None if no <answer> tag is found.
    """
    match = re.search(r"<judgment>([\s\S]*?)</judgment>", predict, re.DOTALL)
    if not match:
        return predict
    return match.group(1).strip()

def azure_gpt4(messages, model):
    if model == "gpt-4o":
        outputs = []
        for message in messages:
            input_prompt = [
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": [  
                        { 
                            "type": "text", 
                            "text": message["instruction"] 
                        },
                        # { 
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": message["image"]
                        #         }
                        # }
                    ]} 
                ]
            ## try N times if API exceed limit ... 
            for i in range(3):
                try:
                    output = client.chat.completions.create(
                        model=model, messages=input_prompt, max_tokens=2000 
                    )

                    output_text = output.choices[0].message.content
                    break ## exit if successful
                
                except Exception as e:
                    print(f'Index {i} got error message: {e}')
                    output_text = ''
                    time.sleep(3)

            outputs.append(output_text)    

        return outputs
    elif model == "o1-mini":
        outputs = []
        for message in messages:
            input_prompt = [
                    # { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": [  
                        { 
                            "type": "text", 
                            "text": message["instruction"] 
                        },
                        # { 
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": message["image"]
                        #         }
                        # }
                    ]} 
                ]
            ## try N times if API exceed limit ... 
            for i in range(10):
                try:
                    output = client.chat.completions.create(
                        model=model, messages=input_prompt, max_completion_tokens=2000 
                    )

                    output_text = output.choices[0].message.content
                    break ## exit if successful
                
                except Exception as e:
                    print(f'Index {i} got error message: {e}')
                    output_text = ''
                    time.sleep(3)

            outputs.append(output_text)    

        return outputs
    else:
        return None


client = AzureOpenAI(
        api_key = "83f30a2a22324395b854bd343db38d85",  
        api_version = "2024-08-01-preview",
        azure_endpoint = "https://francecentral.api.cognitive.microsoft.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
        )

model = "gpt-4o"




prompt_template = '''You are provided a question, a gold answer, and a candidate answer. Your task is to judge the correctness of the candidate answer. Return your judgment enclosed with <judgment> </judgment>.\nQuestion:{Question}\nReference Answer: {Reference}\nCandidate Answer: {Candidate}'''


def infer(prompt_question, reference, candidate):
    prompt_question = prompt_question.replace('<image>', '')
    reference = extract_answer(reference)
    prompt = prompt_template.replace('{Question}', prompt_question).replace('{Reference}', reference).replace('{Candidate}', candidate)
    
    messages = [
            {"instruction": prompt}, 
            ]
    
    # print('Message: ', messages)
    # print('-'*10)
    
    prompt_success = False
    prompt_time = 0
    outputs = ['<judgment> None </judgment>']
    while prompt_success == False and prompt_time <= 3:
        try:
            outputs = azure_gpt4(messages, model)
            prompt_success = True
        except:
            prompt_time += 1
            time.sleep(10)
    
    return outputs[0]



# temp_question = '<image> What is the dominant color in the picture?\nOptions: A: Pink, B: Gray, C: Blue, D: Green'
# temp_reference = '<des> The image features a person riding a unicycle in a park-like setting. The individual is dressed casually in a white shirt, denim shorts, and a white cap, with colorful sneakers. The background is lush with green trees, suggesting a serene outdoor environment. The person is balancing on the unicycle with one leg extended upwards, demonstrating a dynamic pose. The text overlay on the image reads, "Hip and shoulder joints are the most mobile type of joint," and there is a logo for the Spine & Orthopedic Center at the bottom left corner. The overall color palette of the image is dominated by greens from the trees, grays from the pavement, and the white and pink tones from the person\'s attire and sneakers.</des> \n<think> To determine the dominant color in the picture, let\'s analyze the image step-by-step:\n\n1. **Background Analysis**: The background is primarily green due to the trees, which suggests that green is a significant color in the image.\n2. **Foreground Analysis**: The person\'s attire includes white (shirt and cap), denim (shorts), and pink (sneakers). The pavement is gray, and the unicycle has some black components.\n3. **Color Frequency**: Green appears more frequently in the background and less in the foreground, where the person\'s attire and the pavement are more prominent.\n4. **Overall Impression**: While there are multiple colors, green seems to be the most dominant due to its prevalence in the background.\n\nGiven this analysis, the dominant color in the picture is green.</think>\n<answer>4</answer>'
# temp_candidate = 'D'

# print(infer(temp_question, temp_reference, temp_candidate))