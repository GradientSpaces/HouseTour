import torch
import os
import re
import json
import jsonlines
import re

model = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir="/scratch/users/atacelen/.cache/"
os.environ['HF_HOME'] = cache_dir

from transformers import AutoTokenizer
import transformers

access_token = "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"

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=0,
    token=access_token,
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


def eval_q_and_a(question, pred, gt):
    example_output = """
{
    "explanation": "Reasons for the correctness",
    "correctness": 1-5
}
"""

    content = f"""
Given a question, a predicted answer, and a ground truth answer, determine whether the predicted answer is semantically correct compared with the ground truth answer and assign a correctness score from 1 to 5.
Question: {question}
Predicted Answer: {pred}
Ground Truth Answer: {gt}

Example output:
```json
{example_output}
```
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that exclusively outputs responses in valid JSON format. "
                "All responses must be enclosed within triple backticks (```json and ```) to indicate the JSON block. "
                "Ensure the output is properly formatted and adheres to JSON syntax."
            )
        },
        {"role": "user", "content": content},
    ]

    trials = 0
    while trials < 3:
        outputs = pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        response = outputs[0]["generated_text"][-1]["content"]

        match = re.search(r"```json(.*?)```", response, re.DOTALL)
        if match:
            try:
                json_object = json.loads(match.group(1).strip())  # Attempt to parse JSON
                return json_object
            except json.JSONDecodeError:
                trials += 1
        else:
            trials += 1
    
    return None


def eval_q_and_a_bool(question, pred, gt):
    example_output ="""
{
    'explanation' : 'Reasons for the correctness'
    'correct' : true
}
"""

    content =f"""
Given a question, a predicted answer, and a ground truth answer, determine whether the predicted answer is semantically correct compared with the ground truth answer.
Question: {question}
Predicted Answer: {pred}
Ground Truth Answer: {gt}

Example output:
{example_output}
"""
    
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that exclusively outputs responses in valid JSON format. All responses must be enclosed within triple backticks (```json and ```) to indicate the JSON block. Ensure the output is properly formatted and adheres to JSON syntax."},
        {"role": "user", "content": content},
    ]
    
    trials = 0
    while trials < 3:
        outputs = pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        response = outputs[0]["generated_text"][-1]["content"]
        print(response)
        match = re.search(r"```json(.*?)```", response, re.DOTALL)
        if match:
            try:
                json_object = json.loads(match.group(1).strip())  # Attempt to parse JSON
                return json_object
            except json.JSONDecodeError:
                trials += 1
        else:
            trials += 1
    
    return None

def eval_capt(pred, gt):
    example_output ="""
{
    "reference_semantic_elements" : List[str],
    "generated_semantic_elements" : List[str],
    "overlapping_semantic_elements" : List[str]
}
"""

    content =f"""
Identify the semantic elements in the reference and generated texts as well as the overlapping semantic elements

Reference Text: {gt}
Generated Text: {pred}

Example Output:
```json
{example_output}
```
"""
    
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that exclusively outputs responses in valid JSON format. All responses must be enclosed within triple backticks (```json and ```) to indicate the JSON block. Ensure the output is properly formatted and adheres to JSON syntax."},
        {"role": "user", "content": content},
    ]
    
    trials = 0
    while trials < 3:
        outputs = pipeline(
            messages,
            max_new_tokens=4000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        response = outputs[0]["generated_text"][-1]["content"]
        return response
        # match = re.search(r"```json(.*?)```", response, re.DOTALL)
        # if match:
        #     try:
        #         json_object = json.loads(match.group(1).strip())  # Attempt to parse JSON
        #         return json_object
        #     except json.JSONDecodeError:
        #         trials += 1
        # else:
        #     trials += 1
    
    return None


if __name__ == "__main__":        
    data = []
    with jsonlines.open("eval_ours_qa.jsonl", "r") as f:
        for line in f.iter():
            data.append(line)

    llava_next_qa_autodq = []
    
    incorrect, correct = 0, 0
    for d in data:
        result = eval_q_and_a_bool(d['instruction'], d['predicted_answer'], d['ground_truth_answer'])
        if result is None:
            continue
        
        if result['correct']:
            correct += 1
        else:
            incorrect += 1
        
        llava_next_qa_autodq.append(result)
        
        print(f"Rolling Correctness: {correct / (correct + incorrect)}")

    with open('eval_ours_qa_autodq.json', 'w') as f:
        json.dump(llava_next_qa_autodq, f)