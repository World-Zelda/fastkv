from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, set_peft_model_state_dict
import json
import torch
from tqdm import tqdm
import os
import re
import random
from loguru import logger
from safetensors.torch import load_file

def load_and_apply_init_lora_weights(model, init_lora_path):
    logger.info(f"加载初始 LoRA 权重: {init_lora_path}")
    init_lora_weight = load_file(os.path.join(init_lora_path, "adapter_model.safetensors"))

    target_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for name, module in model.named_modules():
        if any(layer in name for layer in target_layers) and "lora" not in name and "base_layer" not in name:
            print(name)
            W = module.weight.data  # 取出原始权重，并移动到 model.device
            
            # 计算初始 LoRA A 和 B
            A_0 = init_lora_weight[name + ".lora_A.weight"].to(W.device).to(W.dtype)
            B_0 = init_lora_weight[name + ".lora_B.weight"].to(W.device).to(W.dtype)
            torch.cuda.empty_cache()
            # 计算 W_res
            W_res = W - B_0 @ A_0
            W_res = W_res.to(W.dtype)
            # print("W:",W)
            # print("B", B_0)
            # print('A', A_0)
            # print("BA:",B_0 @ A_0)
            # print("W_res:",W_res)
            
            # 更新 LoRA 参数
            module.lora_A.data = A_0
            module.lora_B.data = B_0
            
            # 更新残差权重
            module.weight.data = W_res
            
            print(f"Applied SVD to {name}")
    logger.info("成功应用初始 LoRA 权重")
    return model

device = "cuda" # the device to load the model onto

input_file = "/data/xlk/fedsvd/dataset/gsm8k/test.jsonl"
output_file = "/data/xlk/fedsvd/res/gsm8k/LLaMA2-7B/Raw/generate_result.jsonl"

model = AutoModelForCausalLM.from_pretrained(
    "/data/xlk/fedsvd/model/LLaMA3-8B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/data/xlk/fedsvd/model/LLaMA3-8B")
model = PeftModel.from_pretrained(model, "/data/xlk/fedsvd/res/gsm8k/LLaMA3-8B/fedavg/round_2/local_ft", torch_dtype="auto")
# model = load_and_apply_init_lora_weights(model, "/data/xlk/fedsvd/res/gsm8k/Qwen2-7B/federa/init")
# adapters_weights = load_file(os.path.join("/data/xlk/fedsvd/res/gsm8k/Qwen2-7B/federa/round_0/local_ft", "adapter_model.safetensors"))
# set_peft_model_state_dict(model, adapters_weights)
# model = model.merge_and_unload()
# 处理数据
def count_processed_lines(output_path):
    """统计 output_file 已处理的行数"""
    if os.path.exists(output_path):
        return sum(1 for _ in open(output_path, "r", encoding="utf-8"))
    return 0

def process_jsonl_qwen(input_path, output_path):
    processed_lines = count_processed_lines(output_path)
    print(processed_lines)
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "a", encoding="utf-8") as outfile:
        for _ in range(processed_lines):  
            next(infile)  # 跳过已处理的行

        for line in tqdm(infile, desc="Generating..."):
            data = json.loads(line.strip())
            example = "Resolve the issue I provided to you\nHere are a few examples:\n<example>\nquestion: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nanswer:Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72</example>\n<example>question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nanswer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10\n </example>\nThe problem you need to solve is:\n"
            cot = "Let's solve it step by step and provide the answer in the form of 'my answer is:' at the end \n"
            prompt = example + data['question'] + cot 
            # 生成模型输入
            model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
            
            # 生成输出
            with torch.no_grad():
                generated_ids = model.generate(**model_inputs, max_new_tokens=256)
            
            # 解析输出
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 存入数据并写入文件
            data["my_response"] = response
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

def process_jsonl_llama(input_path, output_path):
    processed_lines = count_processed_lines(output_path)
    print(processed_lines)
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "a", encoding="utf-8") as outfile:
        for _ in range(processed_lines):  
            next(infile)  # 跳过已处理的行

        for line in tqdm(infile, desc="Generating..."):
            data = json.loads(line.strip())
            example = "Resolve the issue I provided to you\nHere are a few examples:\n<example>\nquestion: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nanswer:Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72\n</example>\n<example>question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nanswer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10\n </example>\nThe problem you need to solve is:\n"
            cot = "\nLet's solve it step by step and provide the answer in the form of 'my answer is:' at the end \n"
            prompt = example + data['question'] + cot 
            # 生成模型输入
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            # Run the model to infere an output
            outputs = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, top_p=0.95,temperature=0.8)
            response = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
            
            # 存入数据并写入文件
            data["my_response"] = response
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def load_jsonl(fp, instruction, output):
    data_list = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if instruction in data and output in data:
                data_list.append({"instruction": data[instruction], "output": data[output]})
    return data_list

def main():
    list_data_dict = load_jsonl(input_file, instruction='question', output='answer')

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample['instruction'], N_SHOT, COT_FLAG)
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.cuda()
        # Run the model to infere an output
        outputs = model.generate(input_ids=input_ids, max_new_tokens=256, top_p=0.95,temperature=0.8)
        response = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(input_text):]
        model_answer = clean_answer(response)
        is_cor = is_correct(model_answer, sample['output'])
        answers.append(is_cor)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample["instruction"]}\n\n'
              f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
              f'Model Answers: {model_answer}\n\n'
              f'Model Completion: {input_text}\n\n'
              f'Is correct: {is_cor}\n\n')

        print(f'Num of total question: {len(answers)}, '
              f'correct num: {sum(answers)}, '
              f'correct rate: {float(sum(answers))/len(answers)}.')


if __name__ == "__main__":
    main()
