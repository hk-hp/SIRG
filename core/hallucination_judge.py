import os.path
import tqdm
import jsonlines

import json
from openai import OpenAI
import re

api_base = "https://api.chatfire.cn/v1"
api_key = "sk-T5YWsJ8Z3ENuyHjjMBHSietSyIQHA9o5SX64TlNtFWPK6nww"

client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)


def generate(prompt, answer_to_check, model_name="gpt-4o-mini-2024-07-18", timeout=100, retry=10):
    for i in range(retry):
        try:
            completion = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}], timeout=timeout
            )
            content = completion.choices[0].message.content
            print(content)
            answer = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group(0))
            is_correct = answer['is_correct']
            wrong_sents = answer['wrong_sentences']
            if len(wrong_sents) != 0:
                for item in wrong_sents:
                    if item not in answer_to_check:
                        assert 1==2
            return is_correct, wrong_sents
        except:
            continue

    print("No reply from GPT")
    return 'NO', []

hallucination_types = ["IN_CONTEXT", "COUNTER_FACTUAL", "IRRELEVANT", "CORRECT"]

check_correct_prompt_template = """
The question is: {question}
The ground-truth answer is: {gt_answer}
A user has provided a different answer. Please determine if the answer is correct based on the question and the ground-truth answer.
The different answer provided by user is: {answer_to_check}
Is it correct? Return YES or NO.
If you return with NO, please point out the user's wrong sentences. Answer with the original sentence of user.
Answer questions in JSON format: {{"is_correct": "YES/NO", "wrong_sentences": ["sentence1", "sentence2", ...]}}
"""

check_hallucination_type_template = lambda question, answer_to_check: f"""
There are three hallucination type:
COUNTER_FACTUAL. If the answer is inconsistent with factual information (for example, the Earth orbits the Sun, the Sun rises in the East).
IRRELEVANT. If the answer is irrelevant to the question.
IN_CONTEXT. If the answer is inconsistent with the context. Since we do not provide context, you can classify hallucinations that do not fall into the previous two categories as IN_CONTEXT.

The question is: {question}
The hallucinated answer is: {answer_to_check}
Return COUNTER_FACTUAL or IRRELEVANT or IN_CONTEXT according to the type of hallucination.
"""


def judge_if_hallucination_exists(question, gt_answer, answer_to_check):
    prompt = check_correct_prompt_template.format(question=question, 
                                                        gt_answer=gt_answer, 
                                                        answer_to_check=answer_to_check)
    is_correct, wrong_sents = generate(prompt, answer_to_check)
    if "yes" in is_correct.lower():
        return "CORRECT", wrong_sents
    elif "no" in is_correct.lower():
        return "WRONG", wrong_sents
    else:
        return None, None


def sanitize(answer):
    answer = answer.strip()
    if answer.startswith("?") or answer.startswith(",") or answer.startswith(".") or answer.startswith(
            ";") or answer.startswith(":"):
        answer = answer[1:]
    bullshits = ["You are an AI assistant", "You are a helpful assistant", "You are an assistant"]
    for bullshit in bullshits:
        if bullshit in answer:
            answer = answer.split(bullshit)[0]
    self_setting = "(If the question is unanswerable, say \"unanswerable\")"
    if self_setting in answer:
        answer = answer.split(self_setting)[1]
    answer = answer.strip()
    return answer


def judge_hallucination_type(question, answer_to_check):
    gpt_answer = generate(check_hallucination_type_template(question, answer_to_check))
    if "counter_factual" in gpt_answer.lower():
        return "COUNTER_FACTUAL"
    elif "irrelevant" in gpt_answer.lower():
        return "IRRELEVANT"
    elif "in_context" in gpt_answer.lower():
        return "IN_CONTEXT"
    else:
        return "UNKNOWN_TYPE"


if __name__ == '__main__':
    input_file = "data/databricks-dolly-15k_output_3b.jsonl"
    output_file = "data/databricks-dolly-15k_labeled_3b.jsonl"
    contexts = set()
    if os.path.exists(output_file):
        with jsonlines.open(output_file) as context_checker:
            for record in context_checker:
                contexts.add(record["context"])
    try:
        with open(input_file) as f:
            data = json.load(f)
    except:
         data = []
         with jsonlines.open(input_file, 'r') as reader:
            for record in reader:
                data.append(record)

    with jsonlines.open(output_file, 'w') as writer:
        for record in tqdm.tqdm(data):
            if record["context"] in contexts:
                continue
            instruction = record["instruction"]
            gt_answer = record["response"]
            answer_to_check = record["answer"]
            answer_to_check = sanitize(answer_to_check)
            label, wrong_sents = judge_if_hallucination_exists(instruction, gt_answer, answer_to_check)
            if label is None:
                continue
            record["label"] = label
            record["wrong_sents"] = wrong_sents
            """if hallucination_exists is None:
                continue
            elif hallucination_exists:
                hallucination_type = judge_hallucination_type(instruction, answer_to_check)
                record["label"] = hallucination_type
            else:
                record["label"] = "CORRECT"""
            writer.write(record)
