import argparse
import csv
import os
from os.path import join, dirname
#import pandas as pd
import numpy as np
import time
# from dotenv import load_dotenv
from collections import defaultdict

# import boto3
import comet_ml
import pyro
import pyro.distributions as dist
import torch
# from botocore.exceptions import ClientError
from comet_ml import api
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# load_dotenv(verbose=True)
# dotenv_path = join(dirname(__file__), '.env')
# load_dotenv(dotenv_path)

#  revised by LI
def download_posterior(exp_key):
    file_name = exp_key + '.pkl'

    try:
        os.system(f'cp ./pkl_model/{file_name} ./')
    except:
        print(f"Failed to copy {file_name}")


def get_test_data(data_file, test_ids):
    with open(data_file) as f:
        reader = csv.reader(f)

        us = []
        ts = []
        ls = []
        tag_matrix = []

        for i, row in enumerate(reader):
            if i in test_ids:
                us.append(int(row[1]))
                ts.append(int(row[2]))
                ls.append(int(row[3]))
                tag_matrix.append([int(t) for t in row[4].split(",")])

    data = {
        'u': torch.LongTensor(us).to(device),
        't': torch.LongTensor(ts).to(device),
        'l': torch.LongTensor(ls).to(device),
        'tag': torch.LongTensor(tag_matrix).to(device)
    }

    return data


def divide_data_by_user(data, posterior):
    user_count = posterior['gamma_q'].shape[1]
    location_count = posterior['beta_q'].shape[1]

    user_location_matrix = torch.zeros(user_count, location_count).to(device)

    for u, l in zip(data['u'], data['l']):
        user_location_matrix[u][l] += 1

    return user_location_matrix


# No1 new function by Yi
def calculate_prob_for_images(posterior, ids, sample_size):
    # ids : (image_id, loc_id, time_id, uid, items_split)
    # calculate probability of each image, which is from a categorical distribution

    beta_q = posterior['beta_q'].to(device)
    kappa_q = posterior['kappa_q'].to(device)
    delta_q = posterior['delta_q'].to(device)

    size = torch.LongTensor([sample_size]).to(device)

    sigma = dist.Dirichlet(delta_q).sample(size)
    phi = dist.Dirichlet(beta_q).sample(size)
    tau = dist.Dirichlet(kappa_q).sample(size)

    images_prob = torch.zeros(10, len(ids['loc_id']))
    for i in range(0, len(ids['loc_id'])):
        images_prob[:, i] = (phi[:, :, ids['loc_id'][i]] * tau[:, :, ids['time_id'][i]] * (
            sigma[:, :, ids['items_split'][i]]).prod(3)).mean(0)

    return images_prob


# No2 new function by Yi
def calculate_prob_for_loc(posterior, sample_size):
    # ids : (image_id, loc_id, time_id, uid, items_split)
    # calculate probability of each location, which is from a categorical distribution

    beta_q = posterior['beta_q'].to(device)

    size = torch.LongTensor([sample_size]).to(device)

    phi = dist.Dirichlet(beta_q).sample(size)
    loc_prob = torch.zeros(10, len(beta_q[1, :]))
    for i in range(0, len(beta_q[1, :])):
        loc_prob[:, i] = (phi[:, :, i]).mean(0)
    return loc_prob


# No3 new function by Yi
def calculate_scores_for_images(images_prob, n_length, distribution_method):
    # n_length : the number of replication
    # calculate ranking probability, the distribution of ranking for each image
    if distribution_method not in ['uniform', 'normal']:
        return

    a_length = len(images_prob[0, :])
    ranking = torch.zeros(a_length, 100, 10, device=device)
    for j in range(0, 10):
        temp_prob = images_prob[j, :].numpy()
        temp_prob /= sum(temp_prob)
        for i in range(0, n_length):
            temp = torch.tensor(np.random.choice(range(0, a_length), a_length, replace=False, p=temp_prob),
                                device=device)
            for ii in range(1, 101):
                ranking[temp[round(a_length / 100 * (ii - 1)):round(a_length / 100 * ii)], ii - 1, j] += 1
    # ranking : (image_id, image_scores)

    ranking = ranking / n_length
    ranking_final = torch.zeros(a_length, 5, 10, device=device)
    if distribution_method == "uniform":
        ranking_final[:, 4, :] = ranking[:, 0:20, :].sum(1)
        ranking_final[:, 3, :] = ranking[:, 20:40, :].sum(1)
        ranking_final[:, 2, :] = ranking[:, 40:60, :].sum(1)
        ranking_final[:, 1, :] = ranking[:, 60:80, :].sum(1)
        ranking_final[:, 0, :] = ranking[:, 80:100, :].sum(1)

    if distribution_method == "normal":
        ranking_final[:, 4, :] = ranking[:, 0:7, :].sum(1)
        ranking_final[:, 3, :] = ranking[:, 7:31, :].sum(1)
        ranking_final[:, 2, :] = ranking[:, 31:69, :].sum(1)
        ranking_final[:, 1, :] = ranking[:, 69:93, :].sum(1)
        ranking_final[:, 0, :] = ranking[:, 93:100, :].sum(1)

    return ranking_final


# No4 new function by Yi
def calculate_weights_using_scores(ranking, scores, weights_old):
    # scores_from_user : (image_id, scores), weights_old : weights of 10 groups
    # calculate weights of each group using the ranking probability, which is calculated from scores rated by user
    weights = []
    for i in range(0, 10):
        weights.append(weights_old[i] * (ranking[scores[0, :], scores[1, :], i]).prod())

    return weights


# No5 new function by Yi
def recommend_according_to_weights(weights, est_prob, used_for_scores):
    # weights.size : 1*10; est_prob.size : 10*length
    # recommend image/location/activity according to weights and estimated probability
    rec_prob = (torch.tensor(weights).unsqueeze(1) * est_prob).sum(0)
    recommend = torch.argsort(rec_prob, descending=True)
    for ijk in range(0, len(used_for_scores)):
        recommend = torch.cat((recommend[recommend != used_for_scores[ijk]], recommend[recommend == used_for_scores[ijk]]), dim=0)

    return recommend


# No6 new function by Yi
def let_users_give_scores(data):
    high_score_place = list(np.random.choice((data != 0).nonzero().squeeze(), 1, replace=False))
    low_score_place = list(np.random.choice((data == 0).nonzero().squeeze(), 4, replace=False))
    high_score_place.extend(low_score_place)
    high_score = list(np.random.choice(range(3, 5), 1))
    low_score = list(np.random.choice(range(0, 3), 4, replace=True))
    high_score.extend(low_score)
    scores = [high_score_place, high_score]
    scores = torch.tensor(scores)

    return scores


# No7 new function by Yi
def let_user_give_feedback(data, recommend, k):
    scores = []
    recommend_test = recommend.tolist()
    for ijk in range(0, k):
        if data[recommend[ijk]] != 0:
            scores.extend(list(np.random.choice(range(3, 5), 1)))
        else:
            scores.extend(list(np.random.choice(range(0, 3), 1)))
    scores_temp = [recommend_test, scores]
    feedback_info = torch.tensor(scores_temp)

    return feedback_info


# No8 new function by Yi
def evaluation_pre_and_recall(recommend_rank, k, data):
    recommend_top_k = recommend_rank[:, :k].numpy()
    recommend_top_k_bag = torch.zeros(recommend_rank.shape).to(device)
    for i in range(0, k):
        recommend_top_k_bag[range(0, recommend_rank.size(0)), recommend_top_k[:, i]] += 1
    top_k_correct_bag = torch.logical_and(recommend_top_k_bag, data)
    top_k_correct_count_per_user = torch.sum(top_k_correct_bag.type(torch.Tensor), 1).to(device)
    user_count_having_locations = torch.sum(data.sum(1) > 1).to(device)
    precision = torch.sum(top_k_correct_count_per_user) / user_count_having_locations / k
    recalls = top_k_correct_count_per_user / data.sum(1)
    recalls[torch.isnan(recalls)] = 0
    recalls[recalls == float('inf')] = 0
    recall = torch.sum(recalls) / user_count_having_locations
    pre_recall_at_k = {f"location_precision@{k}": precision.to('cpu').item(),
                       f"location_recall@{k}": recall.to('cpu').item()}
    return f"{precision.to('cpu').item()}\t{recall.to('cpu').item()}",{f"location_precision@{k}": precision.to('cpu').item(), f"location_recall@{k}": recall.to('cpu').item()}
#    return pre_recall_at_k


# No1 modified function by Yi
def calc_score(recommend, data_per_user):

    result_metrics = {}

    pre_recall_at_one = evaluation_pre_and_recall(recommend, 1, data_per_user)
    result_metrics.update(pre_recall_at_one[1])
    #print(pre_recall_at_one)

    pre_recall_at_five = evaluation_pre_and_recall(recommend, 5, data_per_user)
    result_metrics.update(pre_recall_at_five[1])
    #print(pre_recall_at_five)

    pre_recall_at_ten = evaluation_pre_and_recall(recommend, 10, data_per_user)
    result_metrics.update(pre_recall_at_ten[1])
    #print(pre_recall_at_ten)
    print(f"{pre_recall_at_one[0]}\t{pre_recall_at_five[0]}\t{pre_recall_at_ten[0]}\t",end = "")
    return result_metrics


def run(ex):
    eid = ex.id
    # filter
    if not ex.get_metrics(metric="duration"):
        # not yet finished
        print('Not finished: {} \n'.format(eid))
        return

    try:
        print(f'start:{eid}\t{",".join(ex.get_tags())}\t',end = "")
        #print('start: ', eid,'\tTags: [ ',', '.join(ex.get_tags()),' ]')

        # download posterior
        download_posterior(eid)

        # No1 Edited by Yi
        posterior = torch.load(eid + '.pkl')
        alpha_q = posterior['alpha_q']
        theta = dist.Dirichlet(alpha_q).sample(torch.LongTensor([10000]))
        weights_ini = theta.sum(0)
        # No1 Edited by Yi

        # prepare test data
        test_data = get_test_data(posterior['data_file'], posterior['test_ids'])
        data_per_user = divide_data_by_user(test_data, posterior)

        # No2 Edited by Yi
        data_per_user_new = data_per_user[((data_per_user != 0).sum(1) > 1), :]

        locs_prob = calculate_prob_for_loc(posterior, 10000)

        ranking_temp = calculate_scores_for_images(locs_prob, 1000, "uniform")

        recommend_first = torch.zeros(data_per_user_new.size(0), locs_prob.size(1))
        weights_all = torch.zeros(data_per_user_new.size(0), locs_prob.size(0))
        recommend_second = torch.zeros(data_per_user_new.size(0), locs_prob.size(1))
        recommend_final = torch.zeros(data_per_user_new.size(0), locs_prob.size(1))

        for iii in range(0, data_per_user_new.size(0)):
            scores_from_user = let_users_give_scores(data_per_user_new[iii, :])
            data_per_user_new[iii, scores_from_user[0, 0]] = 0
            weights_temp = calculate_weights_using_scores(ranking_temp, scores_from_user, weights_ini)
            recommend_temp = recommend_according_to_weights(weights_temp, locs_prob, scores_from_user[0, :])
            recommend_first[iii, :] = recommend_temp
            scores_feedback = let_user_give_feedback(data_per_user_new[iii, :], recommend_temp[0:5], 5)
            weights_second = calculate_weights_using_scores(ranking_temp, scores_feedback, weights_temp)
            recommend_second[iii, :] = recommend_according_to_weights(weights_second, locs_prob, recommend_temp[0:5])
            recommend_final[iii, :] = torch.cat((recommend_first[iii, :5], recommend_second[iii, :(recommend_second.size(1)-5)]), dim=0)

        metrics = calc_score(recommend_first, data_per_user_new)
        # No2 Edited by Yi

        # log update
        ex.log_metrics(metrics)

        # rm pickl
        if os.path.exists(eid + '.pkl'):
            os.remove(eid + '.pkl')
        else:
            print('Could not remove: ', eid + '.pkl')

        print(f'done:{eid}')
        #print('done: ', eid, "\n")
    except:
        print('error: ', eid, "\n")
        return


def main(args):
    print("start_info\ttags\tpre@1\trecal@1\tpre@5\trecal@5\tpre@10\trecal@10\tdone_info")
    my_comet_api = dict(api_key="rAOeE45NqnekTXmKrqg0Do12C",
                        project_name="test-for-pyro",
                        workspace="707728642li",)
    api_key = my_comet_api['api_key']
    workspace_name = my_comet_api['workspace']
    if args.debug:
        project_name = 'test'
    else:
        project_name = my_comet_api['project_name']

    # get experiments
    api_instance = api.API(api_key=api_key)
    q = ((api.Metric('duration') != None) & (api.Parameter('group_count') <= 15))
    exs = api_instance.query(workspace_name, project_name, q)
    # exs = [ ex for ex in exs if "split_by_user" in ex.get_tags()]
    pkl_list = [i.split(".")[0] for i in os.listdir("./pkl_model/") if i.endswith(".pkl")]
    exs = [ ex for ex in exs if ex.id in pkl_list]

    for ex in exs:
        run(ex)


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.1')
    pyro.enable_validation()
    parser = argparse.ArgumentParser(description='pyro model evaluation')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()

    main(args)
