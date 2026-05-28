import json 
from collections import defaultdict
import os 
from matplotlib import category
from tabulate import tabulate 
import re
import sys 
from eval_utils import load_model_results, extract_values_from_json, extract_first_complete_json, model_specific_extraction, model_name_replacement
from math_verify import parse, verify
import re
import string

# ==========================================
# 1. 定义数据集类别 (源自 BBHDatasetRegistry)
# ==========================================

MULTIPLE_CHOICE_SUBSETS = {
    "date_understanding", "disambiguation_qa", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
    "snarks", "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
}

# 提取代码中的 Binary Choice 子集名称 [cite: 213-220]
BINARY_CHOICE_SUBSETS = {
    "boolean_expressions", "causal_judgement", "formal_fallacies",
    "navigate", "sports_understanding", "web_of_lies"
}

# 提取代码中的 Exact Match 子集名称 [cite: 222-226]
EXACT_MATCH_SUBSETS = {
    "multistep_arithmetic_two", "object_counting", "word_sorting"
}

# Dyck 子集 [cite: 228]
DYCK_SUBSETS = {"dyck_languages"}

# ==========================================
# 2. 核心评估函数
# ==========================================

def normalize_text(text: str) -> str:
    """去除首尾空白并转为小写，用于宽松匹配"""
    return text.strip().lower()

def extract_answer_content(model_output: str) -> str:
    """
    尝试从模型输出中提取 'ANSWER:' 之后的内容。
    如果未找到标记，则返回原始输出的最后一行或本身，
    模拟 inspect_ai 中 match(location='end') 的逻辑。
    """
    if "ANSWER:" in model_output:
        # 取最后一个 "ANSWER:" 之后的部分
        return model_output.split("ANSWER:")[-1].strip()
    else:
        # 如果模型没有遵循指令，尝试取最后一行作为备选，或者直接返回去除空白的全文
        return model_output.strip().split('\n')[-1].strip()

def evaluate_bbh_sample(model_answer: str, category: str, target_answer: str) -> bool:
    """
    判断模型回答是否正确。
    
    Args:
        model_answer: 模型生成的原始文本
        category: 题目所属的子集名称 (如 'temporal_sequences')
        target_answer: 数据集中的标准答案 (如 '(A)' 或 'Yes')
    
    Returns:
        True if correct, False otherwise.
    """
    # 1. 预处理：提取模型回答的核心部分
    prediction = extract_answer_content(model_answer)
    
    # 2. 根据类别分发逻辑
    if category in MULTIPLE_CHOICE_SUBSETS:
        return eval_multiple_choice(prediction, target_answer)
    
    elif category in BINARY_CHOICE_SUBSETS:
        return eval_binary_choice(prediction, target_answer)
    
    elif category in EXACT_MATCH_SUBSETS:
        return eval_exact_match(prediction, target_answer)
    
    elif category in DYCK_SUBSETS:
        return eval_exact_match(prediction, target_answer)  # Dyck 使用精确匹配逻辑
    
    else:
        # 未知类别默认使用精确匹配
        return normalize_text(prediction) == normalize_text(target_answer)

# ==========================================
# 3. 具体的匹配逻辑
# ==========================================

def eval_multiple_choice(prediction: str, target: str) -> bool:
    """
    多项选择评估。
    目标通常是 '(A)'。模型可能输出 '(A)', 'A', 'Option (A)' 等。
    """
    # 从目标中提取核心字母，例如 "(A)" -> "A"
    target_match = re.search(r"\(([A-Z])\)", target)
    target_letter = target_match.group(1) if target_match else target.strip()
    
    # 尝试从预测中寻找 "(A)" 形式
    pred_match = re.search(r"\(([A-Z])\)", prediction)
    if pred_match:
        return pred_match.group(1) == target_letter
    
    # 如果找不到括号形式，尝试寻找独立的字母（如 "The answer is A"）
    # 这里使用简单的包含检查作为回退，或者检查去除标点后的结果
    cleaned_pred = prediction.strip(string.punctuation)
    if cleaned_pred == target_letter:
        return True
        
    return False

def eval_binary_choice(prediction: str, target: str) -> bool:
    """
    二元选择评估 (Yes/No, True/False, valid/invalid)。
    使用宽松的词匹配。
    """
    p = normalize_text(prediction)
    t = normalize_text(target)
    
    # 如果提取出的预测非常短，直接比较
    if p == t:
        return True
    
    # 如果预测包含标点 (如 "yes.")，去除标点后比较
    p_clean = p.strip(string.punctuation)
    if p_clean == t:
        return True
        
    return False

def eval_exact_match(prediction: str, target: str) -> bool:
    """
    精确匹配评估 (数字、单词列表、括号序列)。
    """
    # 移除空白后完全一致
    # 注意：word_sorting 要求空格分隔的列表，所以不能简单移除所有空格，只能 strip
    return prediction.strip() == target.strip()

# if __name__ == "__main__":
#     # 模拟用户数据
#     import pandas as pd
    
#     data = {
#         "model_answer": [
#             "Based on the logic, the answer is \nANSWER: (A)", 
#             "The answer is (B).",
#             "ANSWER: Yes",
#             "Calculation result... \nANSWER: 42"
#         ],
#         "category": [
#             "temporal_sequences",      # Multiple Choice
#             "temporal_sequences",      # Multiple Choice
#             "boolean_expressions",     # Binary Choice
#             "object_counting"          # Exact Match
#         ],
#         "answer": [
#             "(A)", 
#             "(A)",
#             "Yes",
#             "42"
#         ]
#     }
    
#     df = pd.DataFrame(data)

#     # 应用函数
#     df['is_correct'] = df.apply(
#         lambda row: evaluate_bbh_sample(row['model_answer'], row['category'], row['answer']), 
#         axis=1
#     )

#     print(df)


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
        cate = item["category"]
        # correct_answer = item["answer"].replace("#", "").strip()
        correct_answer = item["answer"].strip()
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

        first_number_in_model_answer = model_answer
        first_number_in_correct_answer = correct_answer
        # 用mathverifier来验证答案正确与否
        correct = False 
        if first_number_in_model_answer and first_number_in_correct_answer:
            # if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
            gold =first_number_in_correct_answer
            answer =first_number_in_model_answer
            if evaluate_bbh_sample(model_answer=answer,category=cate,target_answer=gold):
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
        parsed_item["model_answer"] = {"raw": raw_model_answer}
        parsed_item["correct_answer"] = {"raw": correct_answer}
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
        #     continue
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
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
 
    data_name = sys.argv[1] if len(sys.argv) > 1 else "math-l5"
    if len(sys.argv) > 1:
        data_name = sys.argv[1]
    run_name_folders = {
        "greedy": f"result_dirs/{data_name}", 
        # "sampling": f"result_dirs/{data_name}/sampling",
        # "greedy@no_cot": f"result_dirs/{data_name}/greedy@no_cot",
    }  
    gen_results(run_name_folders)
