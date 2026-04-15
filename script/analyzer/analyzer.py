# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
from argparse import ArgumentParser
import pandas as pd
import os
import time
import re
from openai import OpenAI
import logging
import shutil

KEY = os.getenv("KEY")
BASE_URL = os.getenv("BASE_URL")


logger = None # info / error / debug
iter_cur = os.getenv("ITER_CUR")
if iter_cur is None:
    iter_cur = '1'

def set_log(args):
    global logger, iter_cur
    
    DATE_TYPE = '%Y.%m.%d-%H.%M.%S'
    time_cur = time.strftime(DATE_TYPE, time.localtime())
    
    logger_path = os.path.join(args.output_path, iter_cur)
    if not os.path.exists(logger_path):
        os.makedirs(logger_path, exist_ok=True)
    
    logger_file = os.path.join(logger_path, f'analyzer_log_{time_cur}_{args.run_type}.log')
    
    logging.basicConfig(filename=logger_file,
                        format='%(asctime)s-%(levelname)s-%(process)d-%(message)s',
                        datefmt=DATE_TYPE,
                        level=logging.INFO)

    logging.Formatter.converter = time.localtime
    
    logger = logging.getLogger()
    logger.handlers.clear()

    file_handler = logging.FileHandler(logger_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s', DATE_TYPE))
    logger.addHandler(file_handler)


def get_args():
    global logger
    
    parser = ArgumentParser()
    
    # for answer question
    parser.add_argument("--prompt-answer-quest", type=str, required=False, 
                        dest='prompt_answer_quest', 
                        default='/workspace/dataset//time_consuming.txt')
    parser.add_argument("--time-consuming-table", type=str, required=False, 
                        dest='time_consuming_table', 
                        default='/workspace/dataset//time_consuming.csv')


    # for give suggestion
    parser.add_argument("--prompt-give-sugg", type=str, required=False, 
                        dest='prompt_give_sugg', 
                        default='/workspace/code/OptiCo/prompt/give_sugg.txt')
    parser.add_argument("--cot-give-sugg", type=str, required=False, 
                        dest='cot_give_sugg', 
                        default='/workspace/code/OptiCo/prompt/give_sugg_cot.txt')
    
    # for change paras
    parser.add_argument("--prompt-change-para", type=str, required=False, 
                        dest='prompt_change_para', 
                        default='/workspace/code/OptiCo/prompt/change_para.txt')
    
    # for locate error
    parser.add_argument("--prompt-loc-error", type=str, required=False, 
                        dest='prompt_loc_error', 
                        default='/workspace/code/OptiCo/prompt/loc_error.txt')
    
    # about monitor
    parser.add_argument("--report-table", type=str, required=False, 
                        dest='report_table', 
                        default='/workspace/code/OptiCo/output/report.csv')
    parser.add_argument("--para-file", type=str, required=False, 
                        dest='para_file', 
                        default='/workspace/code/OptiCo/output/parameters.txt')
    parser.add_argument("--fixed-para-file", type=str, required=False, 
                        dest='fixed_para_file', 
                        default='/workspace/code/OptiCo/output/fixed_parameters.txt')
    

    parser.add_argument("--monitor-result-file", type=str, required=False, 
                        dest='monitor_result_file', 
                        default='/workspace/code/OptiCo/output/monitor_result.txt')
    parser.add_argument("--error-file", type=str, required=False, 
                        dest='error_file', 
                        default='/workspace/code/OptiCo/output/megatron_error.txt')
    
    # common
    parser.add_argument("--output-path", type=str, required=False, 
                        dest='output_path', 
                        default='/workspace/code/OptiCo/output')
    parser.add_argument("--run-type", type=str, required=False, 
                        dest='run_type', 
                        default='auto', 
                        choices=['auto', 'give_sugg', 'change_para', 'loc_error', 'answer_quest']) # auto: monitor --> give_sugg --> change_para --> monitor
    parser.add_argument("--model-path", type=str, required=False, 
                        dest='model_path', 
                        default='/workspace/model/DeepSeek-R1-0528-Qwen3-8B')
    parser.add_argument("--max-length", type=int, required=False, 
                        dest='max_length', 
                        default=8192)
    parser.add_argument("--tp", type=int, required=False, 
                        dest='tp', 
                        default=8)
    parser.add_argument("--pp", type=int, required=False, 
                        dest='pp', 
                        default=1)
    
    args = parser.parse_args()
    
    args.cache_path = os.path.join(args.output_path, 'cache')
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = args.cache_path
    os.environ["HF_HOME"] = args.cache_path

    set_log(args)
    
    # save args    
    logger.info('----------------- arguments start -------------------')
    args_dict = args.__dict__
    for key, value in args_dict.items():
        logger.info(f'\t{key}: {value}')
    logger.info('----------------- arguments end -------------------')

    return args


def read_file(file_path):
    global logger
    
    logger.info(f'read file {file_path}')

    file = open(file_path, 'r')
    ret = file.read()
    file.close()
    return ret


class ChangePrompt:
    def __init__(self, input_text, run_type, args):
        self.input_text = input_text
    
        if run_type == 'give_sugg':
            self.input_prompt = self.give_sugg(args.cot_give_sugg)
        elif run_type == 'change_para':
            self.input_prompt = self.change_para()
        elif run_type == 'loc_error':
            self.input_prompt = self.loc_error()
        elif run_type == 'answer_quest':
            self.input_prompt = self.answer_quest()
    
    def raw(self):
        return "You are a reasoning assistant.\n " + self.input_text

    def give_sugg(self, cot_path=None):
        instruction = (
                        "You are an expert in distributed deep learning performance tuning. Your task is to analyze performance data and propose hyperparameter adjustments to maximize Model FLOPs Utilization (MFU).\n\n"
                        "CRITICAL FORMAT REQUIREMENT: \n"
                        "You MUST ALWAYS provide your reasoning inside `<think>` tags first, then provide your final suggestions outside the think tags. \n"
                        "Your final output must strictly follow this format: \n"
                        "`<think>` \n"
                        "[Your step-by-step reasoning goes here] \n"
                        "`</think>` \n"
                        "The suggestions are: \n"
                        "[Your formatted suggestions here] \n\n"
                       )
        if cot_path is None:
            few_shot_cot = ""
        else:
            cot_text = read_file(cot_path)
            few_shot_cot = (f'{cot_text}\n\n')
        user_prompt = (
                    "Please answer the following open-ended question. "
                    "You should think step by step to solve it.\n\n"
                    "Strictly provide your final answer in the format: \n"
                    "`<think>`\n"
                    "[Your step-by-step reasoning goes here]\n"
                    "`</think>`\n\n"
                    "The suggestions are:\n"
                    "[Your formatted suggestions here]\n\n"
                    "Now, here is the task:\n\n"
                    f"Question:\n{self.input_text}\n\n"
                       )
        return instruction + few_shot_cot + user_prompt

    def change_para(self, cot=None):
        instruction = ("You are a reasoning assistant "
                       "with the ability to modify the parameter configuration according to the user's requirements. \n"
                       )
        
        user_prompt = ('Please answer the following question. '
                       'You should think step by step to solve it. \n\n'
                       'Strictly provide your final answer in the format: '
                       'The modified parameters are: \nYOUR_ANSWER\n\n'
                       f'Question:\n{self.input_text}\n\n'
                       )
        return instruction + user_prompt
    
    def loc_error(self, cot=None):
        instruction = ("You are a reasoning assistant "
                       "with the ability to locate the cause of the error from the logs. \n"
                       )
        user_prompt = ('Please answer the following open-ended question. '
                       'You should think step by step to solve it. \n\n'
                       'Strictly provide your final answer in the format: '
                       'The conclusion is: \nYOUR_ANSWER\n\n'
                       f'Question:\n{self.input_text}\n\n'
                       )
        return instruction + user_prompt

    def answer_quest(self, cot=None):
        instruction = ("You are a reasoning assistant "
                       "with the ability to accurately answer user's questions by analyzing tabular data. \n"
                       # ---------------- few-shot cot start ----------------
                       # ---------------- few-shot cot end ----------------
                       )
        
        user_prompt = ('Please answer the following open-ended question. '
                       'You should think step by step to solve it.\n\n'
                       'Strictly provide your final answer in the format: '
                       'The conclusion is: \nYOUR_ANSWER\n\n'
                       
                       f'Question:\n{self.input_text}\n\n'
                       )
        return instruction + user_prompt


def convert_table2md(report_table):
    global logger
    
    logger.info(f'read table {report_table}')
    
    df = pd.read_csv(report_table, engine='python', encoding='utf-8', index_col=0, na_filter=False)
    ret = df.to_markdown()
    ret = re.sub(r' *\| *', '|', ret)
    ret = re.sub('-+', '-', ret)
    ret = ret.replace('-:','-')
    ret = ret.replace(':-','-')
    return ret

class MergeInput:
    def __init__(self, args, run_type):
        self.prompt_give_sugg = args.prompt_give_sugg
        self.prompt_change_para = args.prompt_change_para
        self.prompt_loc_error = args.prompt_loc_error
        self.report_table = args.report_table
        self.para_file = args.para_file
        self.fixed_para_file = args.fixed_para_file

        self.error_file = args.error_file
        self.output_path = args.output_path


        self.prompt_answer_quest = args.prompt_answer_quest
        self.time_consuming_table = args.time_consuming_table
        
        
        if run_type == 'give_sugg':
            self.input_text = self.give_sugg()
        elif run_type == 'change_para':
            self.input_text = self.change_para()
        elif run_type == 'loc_error':
            self.input_text = self.loc_error()
        elif run_type == 'answer_quest':
            self.input_text = self.answer_quest()

    def answer_quest(self):
        input_text = read_file(self.prompt_answer_quest)
        time_consuming_table = convert_table2md(self.time_consuming_table)
        
        input_text = input_text.replace('{TABLE}', time_consuming_table)
        
        return input_text
    

    def give_sugg(self):
        input_text = read_file(self.prompt_give_sugg)
        fixed_parameters = read_file(self.fixed_para_file)
        report_table = convert_table2md(self.report_table)
        parameters = read_file(self.para_file)
        
        input_text = input_text.replace('{FIXED_PARA}', fixed_parameters)
        input_text = input_text.replace('{TABLE}', report_table)
        input_text = input_text.replace('{PARAMETERS}', parameters)

        return input_text

    def change_para(self):
        input_text = read_file(self.prompt_change_para)
        report_table = convert_table2md(self.report_table)
        parameters = read_file(self.para_file)
        suggestions = read_file(os.path.join(self.output_path, 'suggestion.txt'))

        lines = parameters.strip().split('\n')
        param_names = [line.split(':')[0].strip() for line in lines if line.strip()]
        param_names = ', '.join(param_names)
            
        input_text = input_text.replace('{TABLE}', report_table)
        input_text = input_text.replace('{PARAMETERS}', parameters)
        input_text = input_text.replace('{SUGGESTIONS}', suggestions)
        input_text = input_text.replace('{PARAMETER_NAME}', param_names)
        
        return input_text

    def loc_error(self):
        input_text = read_file(self.prompt_loc_error)
        parameters = read_file(self.para_file)
        fail_log = read_file(self.error_file)
        
        input_text = input_text.replace('{PARAMETERS}', parameters)
        input_text = input_text.replace('{LOG}', fail_log)
        
        return input_text
    
def prepare_input_message(args, run_type):
    global logger
    
    input_text = MergeInput(args, run_type).input_text
    logger.info(f'len(input_text) = {len(input_text)}')

    input_prompt = ChangePrompt(input_text, run_type, args).input_prompt
    logger.info(f'len(input_prompt) = {len(input_prompt)}')

    return input_prompt
    # return [input_message]

def run_vllm(args, run_type):
    global logger
    from vllm import LLM, SamplingParams
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, cache_dir=args.cache_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    input_prompt = prepare_input_message(args, run_type)

    input_message = [{"role": "user", "content": input_prompt}]
    input_message = tokenizer.apply_chat_template(input_message,
            chat_template="{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}",
            tokenize=False,
            add_generation_prompt=True
        )
    
    logger.info(f'input_message is: \n{input_message}')
    
    llm = LLM(model=args.model_path, 
            #   dtype="bfloat16", 
              tensor_parallel_size=args.tp, 
              pipeline_parallel_size=args.pp, 
              gpu_memory_utilization=0.8, 
              enforce_eager=True, 
              max_num_seqs=512)

    sampling_params = SamplingParams(max_tokens=args.max_length, 
                                     temperature=0.6, 
                                     top_p=0.95, 
                                     top_k=20, 
                                     repetition_penalty=1.1, 
                                     seed=1234,
                                     stop=[tokenizer.eos_token, "<|end_search_query|>", "<|end_code_query|>"],
                                     include_stop_str_in_output=True)
    outputs = llm.generate(input_message, sampling_params)
    response = outputs[0].outputs[0].text 
    
    return response

def api_infer(args, run_type, max_retries=10, wait_time=10):
    global logger

    input_prompt = prepare_input_message(args, run_type)
    logger.info(f'input_prompt is: \n{input_prompt}')
    
    client = OpenAI(
        api_key=KEY,
        base_url=BASE_URL,
    )
    for retry_count in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "user", "content": input_prompt}
                ],
                temperature=0.7,
                max_tokens=8192
            )
            return response.choices[0].message.content
        except:
            logger.warning(f"Retry [{retry_count+1}]... waiting {wait_time} sec...")
            time.sleep(wait_time)
    raise Exception("API request failed") 

def read_txt_to_dict(file_path):
    config_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                config_dict[key.strip()] = value.strip()
    return config_dict

def write_dict_to_txt(data, output_path):
    with open(output_path, 'w') as file:
        for key, value in data.items():
            file.write(f"{key}:{value}\n") 


def run_analyzer(args, run_type):
    global logger
    logger.info(f'start to run {run_type}')
    
    time_start = time.time()

    # response = run_vllm(args, run_type)
    response = api_infer(args, run_type)
    logger.info(f'response is: \n{response}')
    
    time_end = time.time()
    time_infer = round(time_end - time_start, 3)
    logger.info(f'infer_time: {time_infer} s')
    
    if run_type == 'give_sugg':    
        # save sugg
        suggestion = response.split('</think>')[-1]
        suggestion = suggestion.split('The suggestions are:')[-1]
        suggestion = suggestion.strip()
        logger.info(f'------------- suggestion is: ------------- \n{suggestion}')
        output_file_suggestion = os.path.join(args.output_path, 'suggestion.txt')
        with open(output_file_suggestion, 'w', encoding='utf-8') as f:
            f.write(suggestion)
            
    elif run_type == 'change_para':
        # save new para
        para_new = response.split('</think>')[-1]
        para_new = para_new.split('The modified parameters are:')[-1]
        para_new = para_new.strip()
        para_new = para_new.replace(',','\n')
        logger.info(f'------------- para_new is: ------------- \n{para_new}')
        para_dic = read_txt_to_dict(args.para_file)
        para_new_dic = dict(re.findall(r'([^:\n]+):([^\n]+)', para_new))
        for key in para_dic.keys():
            if key in para_new_dic:
                para_dic[key] = para_new_dic[key]
        write_dict_to_txt(para_dic, args.para_file)
        
    elif run_type == 'loc_error':
        cause = response.split('</think>')[-1]
        cause = cause.split('The conclusion is:')[-1]
        cause = re.sub(r'\r\n?|\n', '', cause)
        # cause = cause.split('\n')[0]
        cause = cause.strip()
        cause = re.sub(r'\n\s*\n', '\n\n', cause).strip()
        logger.info(f'------------- cause is: ------------- \n{cause}')
        output_file_cause = os.path.join(args.output_path, iter_cur,'loc_error.txt')
        with open(output_file_cause, 'w', encoding='utf-8') as f:
            f.write(cause)
        shutil.copy(output_file_cause, os.path.join(args.output_path,'loc_error.txt'))
        # save fail reason to csv
        df = pd.read_csv(args.report_table, na_filter=False)
        df[df.columns[-1]] = df[df.columns[-1]].astype(object)
        if not df.empty:
            df.iloc[-1, -1] = cause 
        df.to_csv(args.report_table, index=False)

    elif run_type == 'answer_quest':
        answer = response.split('</think>')[-1]
        logger.info(f'------------- answer is: ------------- \n{answer}')


if __name__ == "__main__":
    args = get_args()
    run_type = args.run_type
    if run_type == 'auto':
        run_analyzer(args, 'give_sugg')
        run_analyzer(args, 'change_para')
    else:
        run_analyzer(args, run_type)
    