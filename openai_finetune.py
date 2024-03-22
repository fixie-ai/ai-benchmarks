import os
import time

import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

training_file = "/Users/zachkoch/Downloads/GPT3_5_complete_fine_tuning_examples.jsonl"
training_file_response = client.files.create(
    file=open(training_file, "rb"), purpose="fine-tune"
)
training_file_id = training_file_response.id
print(f"Training file uploaded with ID: {training_file_id}")

fine_tuning_job = client.fine_tuning.jobs.create(
    training_file=training_file_id, model="gpt-3.5-turbo-1106"
)
job_id = fine_tuning_job.id
print(f"Fine-tuning job created with ID: {job_id}")

while True:
    try:
        fine_tuning_status = client.fine_tuning.jobs.retrieve(job_id)
    except openai.InvalidRequestError as e:
        print(e)
        time.sleep(1)
        continue

    status = fine_tuning_status.status
    print(f"Fine-tuning job status: {status}")

    if status in ["succeeded", "failed"]:
        break

    time.sleep(60)
fine_tuned_model_id = fine_tuning_status.fine_tuned_model

completion = client.chat.completions.create(
    model=fine_tuned_model_id,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": "What is 1/2 + 1/4?"},
    ],
)
print(completion.choices[0].message)
