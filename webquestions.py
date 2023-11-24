import json
import tldextract
from pprint import pp
from retrieval_importance import learn_importance, encode_retrievals, encode_groups, v_grouped, \
    most_important_groups, least_important_groups
from retrieval_importance import cal_acc, generate_val_test_set, sort_values, get_retain_urls, cal_acc_reweight, cal_loo, load_openai_retrievals
from retrieval_importance.utils import get_project_root

def utility(retrieval, prediction):
    if prediction in retrieval["correct_answers"]:
        return 1.0
    else:
        return 0.0

def group(retrieved):    
    url_parts = tldextract.extract(retrieved)
    return f'{url_parts.domain}.{url_parts.suffix}'

def experiment_prune(random_seed, retrievals, K = 10, lr = 500, epoch = 50):
    val_set, test_set = generate_val_test_set(len(retrievals), random_seed)
    val_retrievals = [retrievals[i] for i in val_set]
    
    encoded_retrievals, mapping = encode_retrievals(val_retrievals, "retrieved_websites", "retrieved_answers", utility)
    grouping, group_mapping = encode_groups(mapping, group)
    v_ungrouped = learn_importance(encoded_retrievals, k=K, learning_rate=lr, num_steps=epoch, n_jobs=-1, grouping=grouping)
    v = v_grouped(v_ungrouped, grouping, group_mapping)

    v_sorted, total_doc = sort_values(retrievals, val_set, v, group)

    results = [] 
    for remove_rate in range(0, 10, 1):
        retain_urls = get_retain_urls(v_sorted, total_doc, remove_rate/10)
        acc_dev = cal_acc(val_set, retrievals, group, retain_urls, K)
        acc_test = cal_acc(test_set, retrievals, group, retain_urls, K)
        results.append((remove_rate/10, acc_dev, acc_test))

    acc_baseline = results[0][2]

    results.sort(key=lambda x: x[1], reverse=True)
    acc_best = results[0][2]
    threshold = results[0][0]

    return acc_baseline, acc_best, threshold

def experiment_reweight(random_seed, retrievals, K = 10, lr = 500, epoch = 50, threshold = 0.5):
    val_set, test_set = generate_val_test_set(len(retrievals), random_seed)
    val_retrievals = [retrievals[i] for i in val_set]

    encoded_retrievals, mapping = encode_retrievals(val_retrievals, "retrieved_websites", "retrieved_answers", utility)
    grouping, group_mapping = encode_groups(mapping, group)
    v = learn_importance(encoded_retrievals, k=K, learning_rate=lr, num_steps=epoch, n_jobs=-1, grouping=grouping)
    v_per_group = v_grouped(v, grouping, group_mapping)

    keep_dict = {str(i): 1 for i in v_per_group}
    acc_baseline = cal_acc(test_set, retrievals, group, keep_dict, K)
    acc_reweight = cal_acc_reweight(test_set, retrievals, group, group_mapping, v_per_group)
    
    return acc_baseline, acc_reweight

def experiment_loo(random_seed, retrievals, K = 10):
    val_set, test_set = generate_val_test_set(len(retrievals), random_seed)
    val_retrievals = [retrievals[i] for i in val_set]

    v = cal_loo(val_retrievals, group)
    v_sorted, total_doc = sort_values(retrievals, val_set, v, group)

    results = [] 
    for remove_rate in range(0, 10, 1):
        retain_urls = get_retain_urls(v_sorted, total_doc, remove_rate/10)
        acc_dev = cal_acc(val_set, retrievals, group, retain_urls, K)
        acc_test = cal_acc(test_set, retrievals, group, retain_urls, K)
        results.append((remove_rate/10, acc_dev, acc_test))

    acc_baseline = results[0][2]

    results.sort(key=lambda x: x[1], reverse=True)
    acc_best = results[0][2]
    threshold = results[0][0]

    return acc_baseline, acc_best, threshold

def load_retrievals():
    retrievals = []
    with open(f'{str(get_project_root())}/test_data/webquestion.jsonl') as f:
        for line in f:
            retrievals.append(json.loads(line))
    return retrievals

def work_load(metric):
    seed_list = [441, 1, 469, 53, 280, 123, 219, 181, 5, 9, 199, 156, 93, 313, 28, 56, 359, 108, 8, 58, 407, 451, 322, 266, 268, 297, 12, 182, 320, 474, 296, 142, 64, 201, 32, 392, 98, 242, 344, 438, 427, 35, 77, 394, 39, 55, 330, 38, 67, 358, 237, 149, 405, 420, 411, 57, 488, 49, 42, 155, 109, 73, 331, 128]

    retrievals = load_retrievals()

    if metric == "prune":
        result_list = []
        for random_seed in seed_list:
            result_list.append(experiment_prune(random_seed, retrievals))
            print("Finish random seed %d"%(random_seed))
            print(result_list[-1])
    
        acc_baseline = sum([i[0] for i in result_list])/len(result_list)
        acc_prune = sum([i[1] for i in result_list])/len(result_list)
        acc_threshold = sum([i[2] for i in result_list])/len(result_list)
        return acc_baseline, acc_prune, acc_threshold
    
    elif metric == "reweight":
        result_list = []
        for random_seed in seed_list:
            result_list.append(experiment_reweight(random_seed, retrievals))
            print("Finish random seed %d"%(random_seed))
            print(result_list[-1])
    
        acc_baseline = sum([i[0] for i in result_list])/len(result_list)
        acc_reweight = sum([i[1] for i in result_list])/len(result_list)

        return acc_baseline, acc_reweight
    
    elif metric == "loo":
        result_list = []
        for random_seed in seed_list:
            result_list.append(experiment_loo(random_seed, retrievals))
            print("Finish random seed %d"%(random_seed))
            print(result_list[-1])
    
        acc_baseline = sum([i[0] for i in result_list])/len(result_list)
        acc_loo = sum([i[1] for i in result_list])/len(result_list)
        acc_threshold = sum([i[2] for i in result_list])/len(result_list)

        return acc_baseline, acc_loo, acc_threshold

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default="prune", help='loo/reweight/prune')
    args = parser.parse_args()

    with open("./test_data/result/web_question_qa_%s.jsonl"%(args.m), "w") as f:
        if args.m == "prune":
            acc_baseline, acc_prune, acc_threshold = work_load(args.m)
            tmp = {'acc_baseline':acc_baseline, 'acc_prune':acc_prune, 'acc_threshold':acc_threshold}
            print("prune ", acc_baseline, acc_prune, acc_threshold)
            f.write(json.dumps(tmp) + "\n")
            f.flush()       

        elif args.m == "reweight":
            acc_baseline, acc_reweight = work_load(args.m)
            tmp = {'acc_baseline':acc_baseline, 'acc_reweight':acc_reweight}
            print("reweight ", acc_baseline, acc_reweight)
            f.write(json.dumps(tmp) + "\n")
            f.flush()

        elif args.m == "loo":
            acc_baseline, acc_loo, acc_threshold = work_load(args.m)
            tmp = {'acc_baseline':acc_baseline, 'acc_loo':acc_loo, 'acc_threshold':acc_threshold}
            print("loo ", acc_baseline, acc_loo, acc_threshold)
            f.write(json.dumps(tmp) + "\n")
            f.flush()
            