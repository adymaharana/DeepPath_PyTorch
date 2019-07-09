from __future__ import division
import numpy as np
import collections
from itertools import count
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import PolicyNN, ValueNN
from utils import *
from environment import KGEnvironment

# hyperparameters
state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50

dataPath = '../NELL-995/'
model_dir = '../model'
model_name = 'DeepPath'

relation = sys.argv[1]
task = sys.argv[2]
graphpath = os.path.join(dataPath, 'tasks', relation, 'graph.txt')
relationPath = os.path.join(dataPath, 'tasks/', relation, 'train_pos')

class ReinforcementLearningPolicy(nn.Module):

    # TODO: Add regularization to policy neural net and add regularization losses to total loss
    def __init__(self, state_dim, action_space, learning_rate=0.0001):
        super(ReinforcementLearningPolicy, self).__init__()
        self.action_space = action_space
        self.policy_nn = PolicyNN(state_dim, action_space)
        self.optimizer = optim.Adam(self.policy_nn.parameters(), lr=learning_rate)

    def forward(self, state):
        action_prob = self.policy_nn(state)
        return action_prob

    def compute_loss(self, action_prob, target, action):
        # TODO: Add regularization loss
        action_mask = F.one_hot(action, num_classes=self.action_space) > 0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob)*target)
        return loss


def REINFORCE(training_pairs, policy_network, num_episodes):

    train = training_pairs
    success = 0
    done = 0

    # path_found = set()
    path_found_entity = []
    path_relation_found = []

    for i_episode in range(num_episodes):
        start = time.time()
        print('Episode %d' % i_episode)
        print('Training sample: ', train[i_episode][:-1])

        env = KGEnvironment(dataPath, train[i_episode])

        sample = train[i_episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        episode = []
        state_batch_negative = []
        action_batch_negative = []
        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = policy_network(state_vec)
            action_chosen = np.random.choice(np.arange(action_space), p=np.squeeze(action_probs))
            reward, new_state, done = env.interact(state_idx, action_chosen)

            if reward == -1:  # the action fails for this step
                state_batch_negative.append(state_vec)
                action_batch_negative.append(action_chosen)

            new_state_vec = env.idx_state(new_state)
            episode.append(Transition(state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

            if done or t == max_steps:
                break

            state_idx = new_state

        # Discourage the agent when it choose an invalid step
        if len(state_batch_negative) != 0:
            print('Penalty to invalid steps:', len(state_batch_negative))
            predictions = policy_network(np.reshape(state_batch_negative, (-1, state_dim)))
            loss = policy_network.compute_loss(predictions, -0.05, action_batch_negative)
            loss.backward()
            policy_network.optimizer.step()

        print('----- FINAL PATH -----')
        print('\t'.join(env.path))
        print('PATH LENGTH', len(env.path))
        print('----- FINAL PATH -----')

        # If the agent success, do one optimization
        if done == 1:
            print('Success')

            path_found_entity.append(path_clean(' -> '.join(env.path)))

            success += 1
            path_length = len(env.path)
            length_reward = 1 / path_length
            global_reward = 1

            # if len(path_found) != 0:
            # 	path_found_embedding = [env.path_embedding(path.split(' -> ')) for path in path_found]
            # 	curr_path_embedding = env.path_embedding(env.path_relations)
            # 	path_found_embedding = np.reshape(path_found_embedding, (-1,embedding_dim))
            # 	cos_sim = cosine_similarity(path_found_embedding, curr_path_embedding)
            # 	diverse_reward = -np.mean(cos_sim)
            # 	print 'diverse_reward', diverse_reward
            # 	total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward
            # else:
            # 	total_reward = 0.1*global_reward + 0.9*length_reward
            # path_found.add(' -> '.join(env.path_relations))

            total_reward = 0.1 * global_reward + 0.9 * length_reward
            state_batch = []
            action_batch = []
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)

            predictions = policy_network(np.reshape(state_batch, (-1, state_dim)))
            loss = policy_network.compute_loss(predictions, total_reward, action_batch_negative)
            loss.backward()
            policy_network.optimizer.step()
        else:
            global_reward = -0.05
            # length_reward = 1/len(env.path)

            state_batch = []
            action_batch = []
            total_reward = global_reward
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)


            predictions = policy_network(np.reshape(state_batch, (-1, state_dim)))
            loss = policy_network.compute_loss(predictions, total_reward, action_batch_negative)
            loss.backward()
            policy_network.optimizer.step()

            print('Failed, Do one teacher guideline')
            try:
                good_episodes = teacher(sample[0], sample[1], 1, env, graphpath)
                for item in good_episodes:
                    teacher_state_batch = []
                    teacher_action_batch = []
                    total_reward = 0.0 * 1 + 1 * 1 / len(item)
                    for t, transition in enumerate(item):
                        teacher_state_batch.append(transition.state)
                        teacher_action_batch.append(transition.action)

                    predictions = policy_network(np.reshape(state_batch, (-1, state_dim)))
                    loss = policy_network.compute_loss(predictions, 1, action_batch_negative)
                    loss.backward()
                    policy_network.optimizer.step()

            except Exception as e:
                print('Teacher guideline failed')
        print('Episode time: ', time.time() - start)
        print('\n')
    print('Success percentage:', success / num_episodes)

    for path in path_found_entity:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))

    relation_path_stats = collections.Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key=lambda x: x[1], reverse=True)

    f = open(os.path.join(dataPath, 'tasks', relation, 'path_stats.txt'), 'w')
    for item in relation_path_stats:
        f.write(item[0] + '\t' + str(item[1]) + '\n')
    f.close()
    print('Path stats saved')

    return


def retrain():

    # TODO: Fix this - load saved model and optimizer state to Policy_network.policy_nn.
    print('Start retraining')
    policy_network = ReinforcementLearningPolicy(state_dim, action_space)

    f = open(relationPath)
    training_pairs = f.readlines()
    f.close()

    policy_network = torch.load('models/policy_supervised_' + relation)
    print("sl_policy restored")
    episodes = len(training_pairs)
    if episodes > 300:
        episodes = 300
    REINFORCE(training_pairs, policy_network, episodes)
    # save model
    print("Saving model to disk...")
    torch.save(policy_network, os.path.join(model_dir, model_name + '.pt'))
    print('Retrained model saved')

def test():

    f = open(relationPath)
    all_data = f.readlines()
    f.close()

    test_data = all_data
    test_num = len(test_data)

    success = 0
    done = 0

    path_found = []
    path_relation_found = []
    path_set = set()


    policy_network = torch.load('models/policy_retrained' + relation)
    print('Model reloaded')

    if test_num > 500:
        test_num = 500

    for episode in range(test_num):
        print('Test sample %d: %s' % (episode, test_data[episode][:-1]))
        env = KGEnvironment(dataPath, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        transitions = []

        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = policy_network(state_vec)

            action_probs = np.squeeze(action_probs)

            action_chosen = np.random.choice(np.arange(action_space), p=action_probs)
            reward, new_state, done = env.interact(state_idx, action_chosen)
            new_state_vec = env.idx_state(new_state)
            transitions.append(Transition(state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

            if done or t == max_steps_test:
                if done:
                    success += 1
                    print("Success\n")
                    path = path_clean(' -> '.join(env.path))
                    path_found.append(path)
                else:
                    print('Episode ends due to step limit\n')
                break
            state_idx = new_state

        if done:
            if len(path_set) != 0:
                path_found_embedding = [env.path_embedding(path.split(' -> ')) for path in path_set]
                curr_path_embedding = env.path_embedding(env.path_relations)
                path_found_embedding = np.reshape(path_found_embedding, (-1, embedding_dim))
                cos_sim = cosine_similarity(path_found_embedding, curr_path_embedding)
                diverse_reward = -np.mean(cos_sim)
                print('diverse_reward', diverse_reward)
                # total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward
                state_batch = []
                action_batch = []
                for t, transition in enumerate(transitions):
                    if transition.reward == 0:
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)
                # TODO: WUT?? Training in test()
                policy_network.update(np.reshape(state_batch, (-1, state_dim)), 0.1 * diverse_reward, action_batch)
            path_set.add(' -> '.join(env.path_relations))

    for path in path_found:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))

    # path_stats = collections.Counter(path_found).items()
    relation_path_stats = collections.Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key=lambda x: x[1], reverse=True)

    ranking_path = []
    for item in relation_path_stats:
        path = item[0]
        length = len(path.split(' -> '))
        ranking_path.append((path, length))

    ranking_path = sorted(ranking_path, key=lambda x: x[1])
    print('Success percentage:', success / test_num)

    f = open(dataPath + 'tasks/' + relation + '/' + 'path_to_use.txt', 'w')
    for item in ranking_path:
        f.write(item[0] + '\n')
    f.close()
    print('path to use saved')
    return


if __name__ == "__main__":
    if task == 'test':
        test()
    elif task == 'retrain':
        retrain()
    else:
        retrain()
        test()
# retrain()



