from collections import Counter
import argparse
import numpy as np
import logging
import torch
import random
import time
import os
from utils import *

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    fix_seed(args.random_seed)
    print("OPENAI_API_KEY:")

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    sample_n = 2       #   two new + one original = three
    sample_st_n = 3    #   the number of self-consistency sampling
    total = 0
    correct_list = []

    # Translate A sentence into another B
    prompt_another = "Significantly rephrase the following prompt to change its wording and structure while maintaining the same meaning and intent. Ensure that the core problem remains unchanged." + "\n" + "Original prompt:" + \
                       args.cot_trigger + "\n" + "Output:"
    prompt_another = decoder.decode(args, prompt_another, 1, 1, sample_n)

    for i, data in enumerate(dataloader):
        # Answer prediction by generating text ...
        print('*************************')
        print("{}st data".format(i + 1))

        # Prepare question template ...
        x, y = data
        y = y[0].strip()
        xx = "Q: " + x[0] + "\n" + "A:"
        xx = xx + " " + args.cot_trigger

        pre_list = []  # ori and self-consistency

        # original 'prompt' and it's CoT
        z = decoder.decode(args, xx, 1, i, sample_st_n)   # response.choices

        for j in range(len(z)):
            z2 = xx + "\n" + z[j].message.content + "\n" + args.direct_answer_trigger_for_zeroshot_cot + " "
            pred = decoder.decode(args, z2, 1, i, 1)
            pred = pred[0].message.content
            print(z2 + pred)  # print Q + A

            # Clensing of predicted answer ...
            pred = answer_cleansing(args, pred)
            pre_list.append(pred)


        # new 'prompt_1' and it's CoT
        x_another_consis = "Q: " + x[0] + "\n" + "A:" + " " + prompt_another[0].message.content
        zz = decoder.decode(args, x_another_consis, 1, i, sample_st_n)   # response.choices

        for j in range(len(zz)):
            x_another_consis = x_another_consis + "\n" + zz[j].message.content + "\n" + args.direct_answer_trigger_for_zeroshot_cot + " "
            pred_another = decoder.decode(args, x_another_consis, 1, i, 1)
            pred_another = pred_another[0].message.content
            print(x_another_consis + pred_another)  # print Q + A

            # Clensing of predicted another answer ...
            pred_another = answer_cleansing(args, pred_another)
            pre_list.append(pred_another)



        # new 'prompt_2' and it's CoT
        x_another_consis = "Q: " + x[0] + "\n" + "A:" + " " + prompt_another[1].message.content
        zzz = decoder.decode(args, x_another_consis, 1, i, sample_st_n)   # response.choices

        for j in range(len(zzz)):
            x_another_consis = x_another_consis + "\n" + zzz[j].message.content + "\n" + args.direct_answer_trigger_for_zeroshot_cot + " "
            pred_another = decoder.decode(args, x_another_consis, 1, i, 1)
            pred_another = pred_another[0].message.content
            print(x_another_consis + pred_another)  # print Q + A

            # Clensing of predicted another answer ...
            pred_another = answer_cleansing(args, pred_another)
            pre_list.append(pred_another)


        # clear null
        pre_list_copy = pre_list
        pre_list = [element for element in pre_list if element != '']

        if pre_list == []:
            pre_list = pre_list_copy


        # Choose the most frequent answer from the list
        print("======= Final Answer of {}st Data ========".format(i + 1))
        print("pred_list: ", pre_list)

        last_pre = max(pre_list, key=pre_list.count)
        print("pred : {}".format(last_pre))
        print("GT : " + y)
        print('*************************')

        # Checking answer ...
        correct = (np.array([last_pre]) == np.array([y])).sum().item()
        correct_list.append(correct)

        total += 1  # np.array([y]).size(0)

        if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
            break
            # raise ValueError("Stop !!")

        # Current Accuracy:
        accuracy_cur = (sum(correct_list) * 1.0 / total) * 100
        print("Current Accuracy: : {}%".format(accuracy_cur))

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("Total_Accuracy : {}%".format(accuracy))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None,
        help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="gpt3", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"],
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=128,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log2/", help="log directory"
    )

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"

    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    elif args.cot_trigger_no == 15:
        args.cot_trigger = "Break the task into steps, define relationships, and use proper formulas to reach the solution."
    elif args.cot_trigger_no == 16:
        args.cot_trigger = "Break it down step-by-step and solve systematically."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")

    return args


if __name__ == "__main__":
    main()