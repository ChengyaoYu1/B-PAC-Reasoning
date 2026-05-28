import json 
from collections import defaultdict
import os 
from tabulate import tabulate 
import re
import sys 
from eval_utils import load_model_results, extract_values_from_json, extract_first_complete_json, model_specific_extraction, model_name_replacement
from math_verify import parse, verify



def sanitize_math_answers(answer):
    # ignore symbols like $ 
    answer = answer.replace("$", "").strip()
    # ignore the units like miles after the number  
    # remove "," in the number
    answer = answer.replace(",", "")
    # convert fractions to float
    if "/" in answer:
        try:
            answer = str(float(eval(answer)))
        except:
            pass
    return answer

def normalize_latex_format(text):
    if not text:
        return text

    # 1. 统一分数格式：将 \dfrac, \tfrac 替换为 \frac
    text = text.replace(r"\dfrac", r"\frac")
    text = text.replace(r"\tfrac", r"\frac")

    # 2. 统一开根号格式：将 \sqrt2 这种写法转换为 \sqrt{2}
    # 正则逻辑：匹配 \sqrt 后紧跟的一个数字或字母（非 { 开头），给它加上花括号
    text = re.sub(r"\\sqrt\s*([0-9a-zA-Z])", r"\\sqrt{\1}", text)

    # 3. 去除 \left 和 \right (很多库处理不好自适应括号，直接变成普通括号最稳)
    text = text.replace(r"\left", "")
    text = text.replace(r"\right", "")
    
    # 4. 统一空格 (可选)：去掉多余空格，避免 "2 + 2" 和 "2+2" 被判错
    # 注意：这步比较激进，如果你的库依赖空格分词请注释掉
    # text = text.replace(" ", "")

    return text.strip()


def extract_boxed_answer(text):

    if "\\boxed{" not in text:
        return text # 如果没有 boxed，就返回原文，留给后续正则去提取数字
    
    # 找最后一个 \boxed{，因为有时候推理过程也有 box，但答案通常在最后
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return text
    
    # 开始提取
    content = ""
    balance = 0
    started = False
    
    # 从 \boxed{ 后面开始遍历
    for char in text[idx + 7:]: # 7 是 len("\boxed{")
        if char == '{':
            balance += 1
            content += char
        elif char == '}':
            if balance == 0:
                # 找到了匹配的结束括号
                return content
            balance -= 1
            content += char
        else:
            content += char
            
    return content


def eval_model(model, filepath):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0 
    num_total_examples = len(data) 
    no_answer = 0  
    
    reason_lens = []
    parsed_results = [] 
    for item in data:  
        # Read and Parse the prediction from model output
        parsed_item = item.copy()
        prediction_str = item["output"][0] 
        prediction_json = extract_first_complete_json(prediction_str)
        flag_parsed_answer = True
        if prediction_json is None or "answer" not in prediction_json:
            prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
            # print("-")
        if prediction_json is None or "answer" not in prediction_json: 
            try_extracted_answer = model_specific_extraction(model, prediction_str)
            if try_extracted_answer:
                # print(f"Extracted answer from model: {try_extracted_answer}")
                prediction_json["answer"] = try_extracted_answer
            else:
                no_answer += 1 
                flag_parsed_answer = False 
                if False and  "3.1" in model: # used for debugging the format of the output
                    print("--------------------------")
                    print(f"No answer for {item['id']}")
                    print(prediction_str)
                    print(prediction_json)
                    print(correct_answer)
                # continue         
        reason = prediction_json.get("reasoning", "")
        # correct_answer = item["answer"].replace("#", "").strip()
        correct_answer = extract_boxed_answer(item["solution"])
        model_answer = None 
        if not flag_parsed_answer:
            # if "{"+correct_answer+"}" in prediction_str:
            # # Note: we assume the answer is correct if it is in the prediction string
            #     parsed_item["remarks"] = "Correct answer + {} is in the prediction string"
            #     model_answer = correct_answer
            # else:
            parsed_item["model_answer"] = {"raw": None, "sanitized": None, "first_number": None} # not matched 
            parsed_item["correct_answer"] = {"raw": correct_answer}
            parsed_item["matched"] = "No answer extracted"
            parsed_results.append(parsed_item) 
            continue
        else:
            model_answer = str(prediction_json["answer"])
        # sanitize the answers
        raw_model_answer = model_answer[:]
        # model_answer = sanitize_math_answers(model_answer)
        # correct_answer = sanitize_math_answers(correct_answer)
         
        # first_number_in_model_answer = re.search(r"-?\d+(\.\d+)?", model_answer)
        # first_number_in_correct_answer = re.search(r"-?\d+(\.\d+)?", correct_answer)

        first_number_in_model_answer =normalize_latex_format(model_answer)
        first_number_in_correct_answer = normalize_latex_format(correct_answer)
        # 用mathverifier来验证答案正确与否
        correct = False 
        if first_number_in_model_answer and first_number_in_correct_answer:
            # if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
            gold = parse(f"\\boxed{{{first_number_in_correct_answer}}}")
            answer = parse(f"\\boxed{{{first_number_in_model_answer}}}")
            if verify(gold, answer):
                correct = True
                # To debug the correct examples
                if False and "SimPO" in model:
                    if raw_model_answer != correct_answer:
                        print(f"Raw Model Answer: {raw_model_answer}")
                        print(f"Model Answer: {model_answer}, Truth: {correct_answer}")
                        print(f"Extracted from model: {first_number_in_model_answer.group()}, Extracted from truth: {first_number_in_correct_answer.group()}")
                        print("--- correct")
            else:
                # To debug the wrong examples
                if False and "3.1" in model:
                    print(f"Model: {model_answer}, Truth: {correct_answer}")
                    print(f"Extracted from model: {first_number_in_model_answer.group()}, Extracted from truth: {first_number_in_correct_answer.group()}")
                    print("--- incorrect")
        if correct:
            solved_examples += 1
        
        # For Debugging:
        # if item["id"] == "gsm8k-main-test-#2":
        #     print(f"Raw Model Answer: {raw_model_answer}")
        #     print(f"Model Answer: {model_answer}, Truth: {correct_answer}")
        #     print(f"Extracted from model: {first_number_in_model_answer.group()}, Extracted from truth: {first_number_in_correct_answer.group()}") 
        #     print("--- correct" if correct else "--- incorrect")

        # For Debugging:
        if False and "SimPO" in model:
            if not correct:
                print(item["id"], "incorrect")
        reason_lens.append(len(reason))

        parsed_item["reasoning"] = reason
        # parsed_item["model_answer"] = {"raw": raw_model_answer, "sanitized": model_answer, "first_number": first_number_in_model_answer.group() if first_number_in_model_answer else None}
        # parsed_item["correct_answer"] = {"raw": correct_answer, "sanitized": correct_answer, "first_number": first_number_in_correct_answer.group() if first_number_in_correct_answer else None}
        parsed_item["model_answer"] = {"raw": raw_model_answer,"normalized": first_number_in_model_answer}
        parsed_item["correct_answer"] = {"raw": correct_answer, "normalized": first_number_in_correct_answer}
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)


 
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results


def gen_results(run_name_folders): 
    model_results = load_model_results(run_name_folders)

    columns = ["Model", "Mode", "Acc", "No answer", "Total", "Reason Lens"]
    rows = []
    for model_name, filepath in model_results.items(): 
        # print(model_name)
        # if model_name in ["gemini-1.5-flash-exp-0827%greedy"]:
        #     continue\
        if "math" not in model_name:
            continue
        result, parsed_results = eval_model(model_name, filepath) 
        # save the parsed_results to the same filepath with a  new prefix 
        parsed_results_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        # create folders if not exist
        os.makedirs(os.path.dirname(parsed_results_filepath), exist_ok=True)
        # save 
        with open(parsed_results_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)
        rows.append(result)

    # sort the rows by puzzle accuracy
    rows = sorted(rows, key=lambda x: -float(x["Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))
    # print(tabulate(rows, headers=columns, tablefmt="github"))

    # write to markdown file
    banner_header = """
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/4666e72d-4202-4283-8e78-e5ce2b030dcf" alt="zebra_banner" style="width: 69%;" />
</div>


"""
    # with open(f"result_dirs/{data_name}.summary.md", "w") as f:
    #     f.write(banner_header+tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center"))

    # write to json file 
    # with open(f"result_dirs/{data_name}.summary.json", "w") as f:
    #     json.dump(rows, f, indent=2)


if __name__ == "__main__":
    run_name_folders = {
        "greedy": os.environ.get("ZEROEVAL_MATH_RESULT_DIR", "result_dirs/math"),
    }
    gen_results(run_name_folders)
