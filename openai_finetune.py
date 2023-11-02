import os
import time

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
training_file = "/Users/juberti/Downloads/pirate_tune.jsonl"
training_file_response = openai.File.create(
    file=open(training_file, "rb"), purpose="fine-tune"
)
training_file_id = training_file_response["id"]
print(f"Training file uploaded with ID: {training_file_id}")

fine_tuning_job = openai.FineTuningJob.create(
    training_file=training_file_id, model="gpt-3.5-turbo"
)
job_id = fine_tuning_job["id"]
print(f"Fine-tuning job created with ID: {job_id}")

while True:
    try:
        fine_tuning_status = openai.FineTune.retrieve(job_id)
    except openai.error.InvalidRequestError as e:
        print(e)
        time.sleep(1)
        continue

    status = fine_tuning_status["status"]
    print(f"Fine-tuning job status: {status}")

    if status in ["completed", "failed"]:
        break

    time.sleep(60)
fine_tuned_model_id = fine_tuning_status["fine_tuned_model_id"]

completion = openai.ChatCompletion.create(
    model=fine_tuned_model_id,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
print(completion.choices[0].message)
