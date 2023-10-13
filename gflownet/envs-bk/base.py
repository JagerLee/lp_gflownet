"""
Base class of GFlowNet environments
"""
from abc import abstractmethod
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.distributions import Categorical
from torchtyping import TensorType
import pickle
from gflownet.utils.common import set_device, set_float_precision
from pathlib import Path

"""
reward function有四种：power，boltzmann，identity，linear_shift
reward_beta是用于调整奖励函数值大小的参数
power的reward公式为(-1*y/reward_norm)**reward_beta(reward_beta次方)
    reward_norm:用于调整reward大小的参数（仅在power公式中使用）
    如果reward_norm没有给定数字，则reward_norm=reward_norm_std_mult*std(energies)能量标准差的reward_norm_std_mult倍数
    
boltzmann的reward公式为exp(-1*y*reward_beta)（e的-xx次方）

"""
"""
sequence中没有的，需要修改的

state2proxy
statebatch2proxy
statetorch2proxy

reward
reward_batch
reward_torchbatch

proxy2reward
reward2proxy

traj2readable

"""


class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        device="cpu",
        float_precision=32,
        env_id=None,#eg：aptamers
        reward_beta=1,#调整reward大小的参数
        reward_norm=1.0,#调整reward大小的参数
        reward_norm_std_mult=0,#调整reward大小的参数
        reward_func="power",#计算reward的公式选择
        energies_stats=None,
        denorm_proxy=False,
        proxy=None,
        oracle=None,
        # proxy_state_format=None,
        **kwargs,
    ):
        # Device
        if isinstance(device, str):
            self.device = set_device(device)
        else:
            self.device = device
        # Float precision
        self.float = set_float_precision(float_precision)
        # Environment
        self.state = []#初始列表为list形式
        self.done = False
        self.n_actions = 0
        self.id = env_id#eg：aptamers
        self.min_reward = 1e-8
        self.reward_beta = reward_beta
        self.reward_norm = reward_norm
        self.reward_norm_std_mult = reward_norm_std_mult
        self.reward_func = reward_func
        self.energies_stats = energies_stats
        self.denorm_proxy = denorm_proxy
        if oracle is None:  # and proxy is not None:
            self.oracle = proxy
        else:
            self.oracle = oracle
        self._true_density = None
        self._z = None
        self.action_space = []
        self.eos = len(self.action_space)#终止动作对应的索引是动作空间的最后一个索引
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)#logsoftmax函数（softmax激活函数的改良版）
        # Assertions
        assert self.reward_norm > 0
        assert self.reward_beta > 0
        assert self.min_reward > 0

    def copy(self):
        # return an instance of the environment
        return self.__class__(**self.__dict__)

    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm

    @abstractmethod
    def get_actions_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        pass

    #返回动作空间,np.ones(动作空间的长度)，如array([1., 1., 1., 1., 1.])
    def get_fixed_policy_output(self):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy. As a baseline, the fixed policy is uniform over the
        dimensionality of the action space.
        """
        return np.ones(len(self.action_space))#返回一个向量，如动作空间的单词类型为5，返回[1,1,1,1,1]

    #最大轨迹长度
    def get_max_traj_len(
        self,
    ):
        return 1e3

    #输入：state(list)，输出：self.statebatch2proxy([state])
    def state2proxy(self, state: List = None):
        """
        Prepares a state in "GFlowNet format" for the proxy.

        Args
        ----
        state : list
            A state
        """
        if state is None:
            state = self.state.copy()
        return self.statebatch2proxy([state])

    #输入：states（list[list]),输出：np.array(states)
    #将state列表转换为np数组形式（数组比列表内存更小，更方便处理）作为proxy的输入
    def statebatch2proxy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Prepares a batch of states in "GFlowNet format" for the proxy.

        Args
        ----
        state : list
            A state
        """
        return np.array(states)

    #输入：states: TensorType["batch", "state_dim"]，输出：["batch", "state_proxy_dim"]（作为proxy的输入）
    def statetorch2proxy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the proxy.
        """
        return states

##
    def state2oracle(self, state: List = None):
        """
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state : list
            A state
        """
        if state is None:
            state = self.state.copy()
        return state

    def statebatch2oracle(self, states: List[List]):
        """
        Prepares a batch of states in "GFlowNet format" for the oracles
        """
        return states

##这个范围内的函数在sequence中都被改写了，因此可以忽略

    #输入：state，输出：self.proxy2reward(self.proxy(self.state2proxy(state)))
    def reward(self, state=None, done=None):
        """
        Computes the reward of a state
        """
        if done is None:
            done = self.done
        if done:
            return np.array(0.0)
        if state is None:
            state = self.state.copy()
        return self.proxy2reward(self.proxy(self.state2proxy(state)))

    #输入：states（list[list]),输出：rewards[list(done)] = self.proxy2reward(self.proxy(states_proxy)).tolist()
    def reward_batch(self, states: List[List], done=None):
        # Deprecated
        """
        Computes the rewards of a batch of states, given a list of states and 'dones'
        """
        if done is None:
            done = np.ones(len(states), dtype=bool)
        states_proxy = self.statebatch2proxy(states)[list(done), :]
        rewards = np.zeros(len(done))
        if states_proxy.shape[0] > 0:
            rewards[list(done)] = self.proxy2reward(self.proxy(states_proxy)).tolist()
        return rewards

    def reward_torchbatch(
        self, states: TensorType["batch", "state_dim"], done: TensorType["batch"] = None
    ):
        """
        Computes the rewards of a batch of states in "GFlownet format"
        """
        if done is None:
            done = torch.ones(states.shape[0], dtype=torch.bool, device=self.device)
        states_proxy = self.statetorch2proxy(states[done, :])
        reward = torch.zeros(done.shape[0], dtype=self.float, device=self.device)
        if states[done, :].shape[0] > 0:
            with torch.no_grad():
                reward[done] = self.proxy2reward(self.proxy(states_proxy))
        return reward

#上述三个是从state经proxy转换为reward的函数

    #基于proxy的输出值（proxy_vals）返回reward值（基于reward function）
    def proxy2reward(self, proxy_vals):
        """
        Prepares the output of an oracle for GFlowNet: the inputs proxy_vals is
        expected to be a negative value (energy), unless self.denorm_proxy is True. If
        the latter, the proxy values are first de-normalized according to the mean and
        standard deviation in self.energies_stats. The output of the function is a
        strictly positive reward - provided self.reward_norm and self.reward_beta are
        positive - and larger than self.min_reward.
        """
        if self.denorm_proxy:#如果denorm_proxy为True，则proxy_vals需要先进行反归一化
            # TODO: do with torch
            proxy_vals = proxy_vals * (self.energies_stats[1] - self.energies_stats[0]) + self.energies_stats[0]
            # proxy_vals = proxy_vals * self.energies_stats[3] + self.energies_stats[2]
        if self.reward_func == "power":
            return torch.clamp(
                (self.proxy_factor * proxy_vals / self.reward_norm) ** self.reward_beta,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "boltzmann":
            return torch.clamp(
                torch.exp(self.proxy_factor * self.reward_beta * proxy_vals),
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "identity":
            return torch.clamp(
                self.proxy_factor * proxy_vals,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "linear_shift":
            return torch.clamp(
                self.proxy_factor * proxy_vals + self.reward_beta,
                min=self.min_reward,
                max=None,
            )
        else:
            raise NotImplemented

    #将reward转换回proxy值
    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into a (negative) energy or values as returned by
        an oracle.
        """
        if self.reward_func == "power":
            return self.proxy_factor * torch.exp(
                (torch.log(reward) + self.reward_beta * np.log(self.reward_norm))
                / self.reward_beta
            )
        elif self.reward_func == "boltzmann":
            return self.proxy_factor * torch.log(reward) / self.reward_beta
        elif self.reward_func == "identity":
            return self.proxy_factor * reward
        elif self.reward_func == "linear_shift":
            return self.proxy_factor * (reward - self.reward_beta)
        else:
            raise NotImplemented

##
    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the policy
        """
        return states

    def state2policy(self, state=None):
        """
        Converts a state into a format suitable for a machine learning model, such as a
        one-hot encoding.
        """
        if state is None:
            state = self.state
        return state

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Converts a batch of states into a format suitable for a machine learning model,
        such as a one-hot encoding. Returns a numpy array.
        """
        return np.array(states)

    def policy2state(self, state_policy: List) -> List:
        """
        Converts the model (e.g. one-hot encoding) version of a state given as
        argument into a state.
        """
        return state_policy

    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state
        return str(state)

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        return readable
##这部分不需要看，在sequence中被改写

    #[0 1 2] [0 1 3] [0 1 4]]
    def traj2readable(self, traj=None):
        """
        Converts a trajectory into a human-readable string.
        """
        return str(traj).replace("(", "[").replace(")", "]").replace(",", "")

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = []
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : tuple
            Last action performed

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        else:
            parents = []
            actions = []
        return parents, actions

    #输入：policy_outputs: TensorType["n_states", "policy_output_dim"]
    #还需要输入mask_invalid_actions: TensorType["n_states", "policy_output_dim"]，将对应的动作的logits设置为无穷小
    #输出：actions, logprobs即每个state对应的下一步动作和这些动作的logit值大小（经过logsoftmax函数处理后的）
    def sample_actions(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],#如np.zeros((5, 10))每个states对应xx个动作（这里的维度应该是完整动作空间的维度）
        sampling_method: str = "policy",
        mask_invalid_actions: TensorType["n_states", "policy_output_dim"] = None,
        temperature_logits: float = 1.0,
        loginf: float = 1e6,#无穷大
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs. This implementation
        is generally valid for all discrete environments.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)#tensor([0, 1, 2, 3, 4]),代表对应的states
        if sampling_method == "uniform":
            logits = torch.ones(policy_outputs.shape).to(device)#所有的logits都为1（logits代表每个动作的概率）
        elif sampling_method == "policy":
            logits = policy_outputs
            logits /= temperature_logits#还可以进一步采用贪婪算法，对每个值进行修改
        if mask_invalid_actions is not None:
            logits[mask_invalid_actions] = -loginf#mask_invalid_actions为True的位置设置为-loginf
        action_indices = Categorical(logits=logits).sample()#输入的是每个state的prob分布，返回的是选择的logit对应的index（即动作索引）
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]#先对每一个logit进行softmax，然后取对应的logit（数值更新），然后获取每个state的对应索引的更新后的logit值
        # Build actions
        actions = [self.action_space[idx] for idx in action_indices]#根据action_indices获取actions，如[(1,),(4,),(10,)]
        return actions, logprobs

    #基于给定的policy的输出（logits）和实际采取的动作返回logprobs
    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        is_forward: bool,
        actions: TensorType["n_states", 2],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions. This
        implementation is generally valid for all discrete environments.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)
        logits = policy_outputs
        if mask_invalid_actions is not None:
            logits[mask_invalid_actions] = -loginf
        # TODO: fix need to convert to tuple: implement as in continuous
        action_indices = (
            torch.tensor(
                [self.action_space.index(tuple(action.tolist())) for action in actions]
            )
            .to(int)
            .to(device)
        )
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        return logprobs

    #输出：traj_list, traj_actions_list
    #作用：根据输入的一条轨迹，根据其最终state，给出所有可以到达该state的全部轨迹
    def get_trajectories(
        self, traj_list, traj_actions_list, current_traj, current_actions
    ):
        """
        Determines all trajectories leading to each state in traj_list, recursively.

        Args
        ----
        traj_list : list
            List of trajectories (lists)
        
        traj_actions_list : list
            List of actions within each trajectory

        current_traj : list
            Current trajectory

        current_actions : list
            Actions of current trajectory

        Returns
        -------
        traj_list : list
            List of trajectories (lists)

        traj_actions_list : list
            List of actions within each trajectory
        """
        parents, parents_actions = self.get_parents(current_traj[-1], False)#对于当前轨迹的最后一个state（即当前state），获取其所有可能父节点，和父节点到当前state的对应动作
        if parents == []:#当前traj的最后一个state也是全为初始pad的souce state（即这个轨迹的所有state都是source state）
            traj_list.append(current_traj)
            if hasattr(self, "action_pad_length"):#如果有action_pad_length属性，则将action的长度补齐
                # Required for compatibility with mfenv when length(sfenv_action) != length(fidelity.action)
                # For example, in AMP, length(sfenv_action) = 1 like (2,), length(fidelity.action) = 2 like (22, 1)
                current_actions = [
                    tuple(list(action) + [0] * (self.action_max_length - len(action)))
                    for action in current_actions
                ]
            traj_actions_list.append(current_actions)
            return traj_list, traj_actions_list
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            traj_list, traj_actions_list = self.get_trajectories(
                traj_list, traj_actions_list, current_traj + [p], current_actions + [a]
            )
        return traj_list, traj_actions_list

    def step(self, action_idx):
        """
        Executes step given an action.

        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action_idx : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if action < self.eos:
            self.done = False
            valid = True
        else:
            self.done = True
            valid = True
            self.n_actions += 1
        return self.state, action, valid

    def no_eos_mask(self, state=None):
        """
        Returns True if no eos action is allowed given state
        """
        if state is None:
            state = self.state
        return False

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if forward action is
        invalid given the current state, False otherwise.
        #
        """
        mask = [False for _ in range(len(self.action_space))]
        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        Returns a vector with the length of the discrete part of the action space + 1:
        True if action is invalid going backward given the current state, False
        otherwise.
        """
        if parents_a is None:
            _, parents_a = self.get_parents()
        mask = [True for _ in range(len(self.action_space))]
        for pa in parents_a:
            mask[self.action_space.index(pa)] = False
        return mask

    def set_state(self, state, done):
        """
        Sets the state and done of an environment.
        """
        self.state = state
        self.done = done
        return self

    def true_density(self):
        """
        Computes the reward density (reward / sum(rewards)) of the whole space

        Returns
        -------
        Tuple:
          - normalized reward for each state
          - un-normalized reward
          - states
        """
        return (None, None, None)

    @staticmethod
    def np2df(*args):
        """
        Args
        ----
        """
        return None


class Buffer:
    """
    Implements the functionality to manage various buffers of data: the records of
    training samples, the train and test data sets, a replay buffer for training, etc.
    """
    #Buffer有两种，一个是main，一个是replay（缓冲区）
    def __init__(
        self,
        env,
        make_train_test=False,
        replay_capacity=0,
        output_csv=None,
        data_path=None,
        train=None,
        test=None,
        logger=None,
        **kwargs,
    ):
        self.logger = logger
        self.env = env
        self.replay_capacity = replay_capacity
        self.main = pd.DataFrame(columns=["state", "traj", "reward", "energy", "iter"])
        self.replay = pd.DataFrame(
            np.empty((self.replay_capacity, 5), dtype=object),
            columns=["state", "traj", "reward", "energy", "iter"],
        )
        self.replay.reward = pd.to_numeric(self.replay.reward)
        self.replay.energy = pd.to_numeric(self.replay.energy)
        self.replay.reward = [-1 for _ in range(self.replay_capacity)]
        # Define train and test data sets
        if train is not None and "type" in train:
            self.train_type = train.type
        else:
            self.train_type = None
        self.train, dict_tr, train_stats = self.make_data_set(train)
        if (
            self.train is not None
            and "output_csv" in train
            and train.output_csv is not None
        ):
            self.train.to_csv(train.output_csv)
        if (
            dict_tr is not None
            and "output_pkl" in train
            and train.output_pkl is not None
        ):
            with open(train.output_pkl, "wb") as f:
                pickle.dump(dict_tr, f)
                self.train_pkl = train.output_pkl
        else:
            print(
                """
            Important: offline trajectories will NOT be sampled. In order to sample
            offline trajectories, the train configuration of the buffer should be
            complete and feasible and an output pkl file should be defined in
            env.buffer.train.output_pkl.
            """
            )
            self.train_pkl = None
        if test is not None and "type" in test:
            self.test_type = test.type
        else:
            self.train_type = None
        self.test, dict_tt, test_stats = self.make_data_set(test)
        if (
            self.test is not None
            and "output_csv" in test
            and test.output_csv is not None
        ):
            self.test.to_csv(test.output_csv)
        if dict_tt is not None and "output_pkl" in test and test.output_pkl is not None:
            with open(test.output_pkl, "wb") as f:
                pickle.dump(dict_tt, f)
                self.test_pkl = test.output_pkl
        else:
            print(
                """
            Important: test metrics will NOT be computed. In order to compute
            test metrics the test configuration of the buffer should be complete and
            feasible and an output pkl file should be defined in
            env.buffer.test.output_pkl.
            """
            )
            self.test_pkl = None
        # Compute buffer statistics
        if self.train is not None:
            (
                self.mean_tr,
                self.std_tr,
                self.min_tr,
                self.max_tr,
                self.max_norm_tr,
            ) = train_stats[0], train_stats[1], train_stats[2], train_stats[3], train_stats[4]
        if self.test is not None:
            self.mean_tt, self.std_tt, self.min_tt, self.max_tt, _ = test_stats[0], test_stats[1], test_stats[2], test_stats[3], test_stats[4]

    def add(
        self,
        states,
        trajs,
        rewards,
        energies,
        it,
        buffer="main",
        criterion="greater",
    ):
        if buffer == "main":
            self.main = pd.concat(
                [
                    self.main,
                    pd.DataFrame(
                        {
                            "state": [self.env.state2readable(s) for s in states],#列值：产物smiles
                            "traj": [self.env.traj2readable(p) for p in trajs],
                            "reward": rewards,
                            "energy": energies,
                            "iter": it,
                        }
                    ),
                ],
                axis=0,
                join="outer",
            )
        elif buffer == "replay" and self.replay_capacity > 0:
            if criterion == "greater":
                self.replay = self._add_greater(states, trajs, rewards, energies, it)#不断更新缓冲区的轨迹（将具有高reward值的轨迹引入，替换掉低reward值的轨迹，直到缓冲区满）

    #不断更新缓冲区的轨迹（将具有高reward值的轨迹引入，替换掉低reward值的轨迹，直到缓冲区满）
    def _add_greater(
        self,
        states,
        trajs,
        rewards,
        energies,
        it,
    ):
        rewards_old = self.replay["reward"].values
        rewards_new = rewards.copy()
        while np.max(rewards_new) > np.min(rewards_old):#不断更新缓冲区的轨迹（将具有高reward值的轨迹引入，替换掉低reward值的轨迹，直到缓冲区满）
            idx_new_max = np.argmax(rewards_new)#获取新的reward的最大值的索引
            readable_state = self.env.state2readable(states[idx_new_max])
            if not self.replay["state"].isin([readable_state]).any():#如果新的reward的最大值对应的state不在replay中
                self.replay.iloc[self.replay.reward.argmin()] = {
                    "state": self.env.state2readable(states[idx_new_max]),
                    "traj": self.env.traj2readable(trajs[idx_new_max]),
                    "reward": rewards[idx_new_max],
                    "energy": energies[idx_new_max],
                    "iter": it,
                }#将缓冲区原来的最小reward对应的轨迹替换为新的reward的最大值对应的轨迹
                rewards_old = self.replay["reward"].values#更新rewards_old
            rewards_new[idx_new_max] = -1#更新后，将对应索引的reward值设置为-1，下次循环时就不会再次被选中
        return self.replay

    #基于config读取数据csv，return df, {"x": samples, "energy": energies}, stats
    def make_data_set(self, config):
        """
        Constructs a data set as a DataFrame according to the configuration.
        """
        stats = None
        if config is None:
            return None, None, None
        elif "path" in config and config.path is not None:
            path = self.logger.logdir / Path("data") / config.path#获取数据集的路径
            df = pd.read_csv(path, index_col=0)
            samples = [self.env.readable2state(s) for s in df["samples"].values]
            stats = self.compute_stats(df)#统计
        elif "type" not in config:
            return None, None, None
        elif config.type == "all" and hasattr(self.env, "get_all_terminating_states"):
            samples = self.env.get_all_terminating_states()
        elif (
            config.type == "grid"
            and "n" in config
            and hasattr(self.env, "get_grid_terminating_states")
        ):
            samples = self.env.get_grid_terminating_states(config.n)
        elif (
            config.type == "uniform"
            and "n" in config
            and hasattr(self.env, "get_uniform_terminating_states")
        ):
            samples = self.env.get_uniform_terminating_states(config.n)
        else:
            return None, None, None
        energies = self.env.proxy(self.env.statebatch2proxy(samples)).tolist()
        df = pd.DataFrame(
            {
                "samples": [self.env.state2readable(s) for s in samples],
                "energies": energies,
            }
        )
        if stats is None:
            stats = self.compute_stats(df)
        return df, {"x": samples, "energy": energies}, stats

    #data的统计分析，如mean，std，min，max，max_norm能量值
    def compute_stats(self, data):
        mean_data = data["energies"].mean()
        std_data = data["energies"].std()
        min_data = data["energies"].min()
        max_data = data["energies"].max()
        data_zscores = (data["energies"] - mean_data) / std_data
        max_norm_data = data_zscores.max()
        return mean_data, std_data, min_data, max_data, max_norm_data

    def sample(
        self,
    ):
        pass

    def __len__(self):
        return self.capacity

    @property
    def transitions(self):
        pass

    def save(
        self,
    ):
        pass

    @classmethod
    def load():
        pass

    @property
    def dummy(self):
        pass
