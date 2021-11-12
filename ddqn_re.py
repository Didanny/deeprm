import time
# import threading
import numpy as np
# import theano
# import cPickle
import pickle
import matplotlib.pyplot as plt
import random

from multiprocessing import Process
from multiprocessing import Manager

import environment
import job_distribution
from parameters import Parameters
# import pg_network
import slow_down_cdf
from replay_memory import ReplayMemory
from q_network import QNetwork

class DDQNAgent:
    def __init__(self, pa) -> None:
        self.memory = ReplayMemory()
        self.gamma = pa.discount # discount rate
        # self.epsilon = 1.0 # exploration rate
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.999
        self.batch_size = pa.batch_size
        self.train_start = 500_000
        self.action_size = pa.num_nw

        self.TAU = 0.1 # target network soft update hyperparameter

        self.online = QNetwork(pa)
        self.target = QNetwork(pa)

    def update_target_model(self):
        self.target.model.set_weights(self.online.model.get_weights())

    def save_model(self, filename):
        self.online.save_model(filename)
    
    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

        # if len(self.memory) > self.train_start:
        #     if self.epsilon > self.epsilon_min:
        #         self.epsilon *= self.epsilon_decay

    def act(self, state):
        # Explore
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        # Exploit
        else:
            return np.argmax(self.online.model.predict(state))

def _replay(online, target_model, replay_memory, pa):
    if replay_memory.__len__() < pa.train_start:
        print("Memory len: {0}".format(replay_memory.__len__()))
        return False, False

    minibatch = replay_memory.sample(pa.batch_size)

    state = np.zeros((pa.batch_size, pa.network_input_height * pa.network_input_width))
    next_state = np.zeros((pa.batch_size, pa.network_input_height * pa.network_input_width))
    action, reward, done = [], [], []

    for i in range(pa.batch_size):
        state[i] = minibatch[i][0]
        action.append(minibatch[i][1])
        reward.append(minibatch[i][2])
        next_state[i] = minibatch[i][3]
        done.append(minibatch[i][4])

    target = online.model.predict(state.reshape(-1, 20*124))
    target_next = online.model.predict(next_state.reshape(-1, 20*124))
    target_val = target_model.model.predict(next_state.reshape(-1, 20*124))

    for i in range(len(minibatch)):
        # correction on the Q value for the action used
        if done[i]:
            target[i][action[i]] = reward[i]
        else:
            # current Q Network selects the action
            # a'_max = argmax_a' Q(s', a')
            a = np.argmax(target_next[i])
            # target Q Network evaluates the action
            # Q_max = Q_target(s', a'_max)
            target[i][action[i]] = reward[i] + pa.discount * (target_val[i][a])

    return state, target

    # self.online.model.fit(state, target, batch_size=self.batch_size, verbose=0)

def _concatenate_all_ob(trajs, pa):

    timesteps_total = 0
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros(
        (timesteps_total, 1, pa.network_input_height, pa.network_input_width),
        dtype=np.float32)

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, 0, :, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob

def _discount(x, gamma):
    # Given vector x, computes a vector y such that
    # y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...

    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

def _process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len

def _concatenate_all_ob_across_examples(all_ob, pa):

    num_ex = len(all_ob)
    total_samp = 0
    for i in range(num_ex):
        total_samp += all_ob[i].shape[0]

    all_ob_contact = np.zeros(
        (total_samp, 1, pa.network_input_height, pa.network_input_width),
        dtype=np.float32)

    total_samp = 0

    for i in range(num_ex):
        prev_samp = total_samp
        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp : total_samp, :, :, :] = all_ob[i]

    return all_ob_contact

def _plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(mean_rew_lr_curve, linewidth=2, label='DQN mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    ax.plot(max_rew_lr_curve, linewidth=2, label='DQN max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(slow_down_lr_curve, linewidth=2, label='DQN mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)

    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")

def _get_traj_test(learner, env, episode_max_length, result):
    # Run agent-environment loop for one whole episode (trajectory)
    # Return dictionary of results

    env.reset()
    obs = []
    acts = []
    rews = []
    entropy = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):
        # act_prob = agent.get_one_act_prob(ob)
        # csprob_n = np.cumsum(act_prob)
        # a = (csprob_n > np.random.rand()).argmax()

        state = ob
        # print("STATE SHAPE: {0}".format(state.shape))
        # print("Prediction: {0}".format(self.online.model.predict(state.reshape(-1, 20, 124))))
        a = np.argmax(learner.model.predict(state.flatten().reshape(-1, 20*124)), axis=-1)[0]
        # print("action-{0}|size-{1}".format(a, self.action_size))
        action = a

        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        ob, rew, done, info = env.step(a, repeat=True)

        reward = rew
        next_state = ob

        rews.append(rew)
        
        result.append((state.flatten(), action, reward, next_state.flatten(), done))

        if done: 
            break

    return {
        'reward': np.array(rews),
        'ob' : np.array(obs),
        'action' : np.array(acts),
        'info' : info
    }

def _get_traj_train(env, episode_max_length, result):
    # Run agent-environment loop for one whole episode (trajectory)
    # Return dictionary of results

    env.reset()
    obs = []
    acts = []
    rews = []
    entropy = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):
        # act_prob = agent.get_one_act_prob(ob)
        # csprob_n = np.cumsum(act_prob)
        # a = (csprob_n > np.random.rand()).argmax()

        state = ob
        a = random.randrange(6)
        action = a

        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        ob, rew, done, info = env.step(a, repeat=True)

        reward = rew
        next_state = ob

        rews.append(rew)
        
        result.append((state.flatten(), action, reward, next_state.flatten(), done))

        if done: break

    return {
        'reward': np.array(rews),
        'ob' : np.array(obs),
        'action' : np.array(acts),
        'info' : info
    }

def _get_traj(weights, env, pa, result, memory_result):
# def _get_traj(learner):
    # print("Hello")
    # return
    trajs = []

    learner = QNetwork(pa)
    learner.model.set_weights(weights)

    for i in range(pa.num_seq_per_batch):
        _get_traj_train(env, pa.episode_max_length, memory_result)
        traj = _get_traj_test(learner, env, pa.episode_max_length, memory_result)
        trajs.append(traj)

    all_ob = _concatenate_all_ob(trajs, pa)

    # Compute discounted sums of rewards
    rets = [_discount(traj["reward"], pa.discount) for traj in trajs]
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

    # Compute time-dependent baseline
    baseline = np.mean(padded_rets, axis=0)

    # Compute advantage function
    # advs = [ret - baseline[:len(ret)] for ret in rets]
    all_action = np.concatenate([traj["action"] for traj in trajs])
    # all_adv = np.concatenate(advs)

    all_eprews = np.array([_discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
    all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths

    # All Job Stat
    enter_time, finish_time, job_len = _process_all_info(trajs)
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

    # all_entropy = np.concatenate([traj["entropy"] for traj in trajs])

    result.append({
        "all_ob": all_ob,
        "all_action": all_action,
        "all_eprews": all_eprews,
        "all_eplens": all_eplens,
        "all_slowdown": all_slowdown
    })
    
def launch(pa, ql_resume=False, render=False, repre='image', end='no_new_job'):
    
    online = QNetwork(pa)
    target = QNetwork(pa)
    envs = []
    replay_memory = ReplayMemory()

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)

    for ex in range(pa.num_ex):

        print("-prepare for env-{0}".format(ex))

        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs,
                            render=False, repre=repre, end=end)
        env.seq_no = ex
        envs.append(env)

    # for ex in range(10 + 1):  # last worker for updating the parameters

    #     print("-prepare for worker-{0}".format(ex))

    #     dqn_learner = QNetwork(pa)

    #     # if pg_resume is not None:
    #     #     net_handle = open(pg_resume, 'rb')
    #     #     net_params = pickle.load(net_handle)
    #     #     pg_learner.set_net_params(net_params)

    #     dqn_learners.append(dqn_learner)

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------

    ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre=repre, end=end)
    
    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    for iteration in range(1, pa.num_epochs):

        ps = []  # threads
        manager = Manager()  # managing return results
        manager_result = manager.list([])
        manager_memory = Manager()
        manager_memory_result = manager_memory.list([])

        ex_indices = [x for x in range(pa.num_ex)]
        np.random.shuffle(ex_indices)
        
        all_eprews = []
        grads_all = []
        loss_all = []
        eprews = []
        eplens = []
        all_slowdown = []
        all_entropy = []

        # go through all examples
        ex_counter = 0
        for ex in range(pa.num_ex):
            ex_idx = ex_indices[ex]
            weights = online.model.get_weights()
            p = Process(target=_get_traj,
                        args=(weights, envs[ex_idx], pa, manager_result, manager_memory_result, ))
            # p = Process(target=_get_traj,
                        # args=(dqn_learners[ex_counter], envs[ex_idx], pa, manager_result, manager_memory_result, ))
            ps.append(p)

            # print("Collecting trajectories...")
            # Collect trajectories until we get timesteps_per_batch total timesteps
            # trajs = []

            ex_counter += 1

            if ex_counter >= 10 or ex == pa.num_ex - 1:

                print("{0} out of {1}".format(ex, pa.num_ex))

                ex_counter = 0

                for p in ps:
                    p.start()

                for p in ps:
                    p.join()

                result = []  # convert list from shared memory
                for r in manager_result:
                    result.append(r)

                for m in manager_memory_result:
                    replay_memory.append(m)

                ps = []
                manager_result = manager.list([])

                # all_ob = _concatenate_all_ob_across_examples([r["all_ob"] for r in result], pa)
                # all_action = np.concatenate([r["all_action"] for r in result])
                # all_adv = np.concatenate([r["all_adv"] for r in result])

                # Do policy gradient update step, using the first agent
                # put the new parameter in the last 'worker', then propagate the update at the end
                # grads = pg_learners[pa.batch_size].get_grad(all_ob, all_action, all_adv)

                # grads_all.append(grads)

                all_eprews.extend([r["all_eprews"] for r in result])

                eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
                eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths

                all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in result]))
                # all_entropy.extend(np.concatenate([r["all_entropy"] for r in result]))

        # TODO: TRAIN MODEL HERE
        s, t = _replay(online, target, replay_memory, pa)
        if s is not False:
            online.model.fit(s, t, batch_size=pa.batch_size, verbose=0)

        weights = online.model.get_weights()
        for i in range(11):
            online.model.set_weights(weights)
            
        timer_end = time.time()

        print("-----------------")
        print("Iteration: \t %i" % iteration)
        print("NumTrajs: \t %i" % len(eprews))
        print("NumTimesteps: \t %i" % np.sum(eplens))
        # print "Loss:     \t %s" % np.mean(loss_all)
        print("MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews]))
        print("MeanRew: \t %s +- %s" % (np.mean(eprews), np.std(eprews)))
        print("MeanSlowdown: \t %s" % np.mean(all_slowdown))
        print("MeanLen: \t %s +- %s" % (np.mean(eplens), np.std(eplens)))
        # print("MeanEntropy \t %s" % (np.mean(all_entropy)))
        print("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
        print("-----------------")

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(np.mean(eprews))
        slow_down_lr_curve.append(np.mean(all_slowdown))

        if iteration % pa.output_freq == 0:
            # param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
            # pickle.dump(pg_learners[pa.batch_size].get_params(), param_file, -1)
            # param_file.close()

            # TODO: SAVE MODEL & UPDATE TARGET

            weights = online.model.get_weights()
            target.model.set_weights(weights)
            online.model.save_weights(pa.output_filename + '_' + str(iteration))    

            # pa.unseen = True
            # slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.pkl',
                                #  render=False, plot=True, repre=repre, end=end)
            # pa.unseen = False
            # test on unseen examples

            _plot_lr_curve(pa.output_filename,
                        max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                        ref_discount_rews, ref_slow_down)

            
if __name__ == '__main__':
    pa = Parameters()

    pa.simu_len = 50
    pa.num_ex = 10
    pa.output_filename = 'ckpt/dqn_re'
    pa.batch_size = 100
    pa.train_start = 1000

    launch(pa)