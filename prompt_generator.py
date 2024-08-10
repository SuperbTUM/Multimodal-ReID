import glob
import base64
import requests
import random
import time
from collections import defaultdict

# OpenAI API Key
api_key = ""


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
base_path = "./Market1501/bounding_box_train"
image_paths = glob.glob("/".join([base_path, "*.jpg"]))

image_list = defaultdict(list)

for image_path in image_paths:
    image_name = image_path.split("/")[-1]
    label = int(image_name.split("_")[0])
    image_list[label].append(image_path)

assert len(image_list) < 1000

with open("prompts_market1501.txt", "w+") as f:
    for label in image_list:

        level = 0
        model = "gpt-4o-mini"

        content = "sorry"

        while "sorry" in content or "unable" in content:

            image_path, image_path2 = random.sample(image_list[label], 2)

            # Getting the base64 string
            base64_image = encode_image(image_path)
            base64_image2 = encode_image(image_path2)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Focus on the person in the photos. Summarize the common parts of the person's clothing and exclude behavior in one sentence starting with 'A photo of a'."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image2}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 256
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            res = response.json()

            while "error" in res:
                time.sleep(1.0)
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                res = response.json()

            content = res["choices"][0]["message"]["content"]

            level += 1

            if level >= 2:
                model = "gpt-4o"

        f.write(str(label) + ":" + content + "\n")
