from abc import abstractmethod
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.distributions import Categorical#概率分布采样对象
from torchtyping import TensorType
import pickle
from pathlib import Path
from gflownet.utils.common import set_device, set_float_precision
from gflownet.utils.huffman import HuffmanTree
import json
import gzip
with gzip.GzipFile('/home/lishuwang/project/GFlowNet/lp_gflownet/gflownet/data/count.json.gz','r') as f_in:
    count=json.load(f_in)
with gzip.GzipFile('/home/lishuwang/project/GFlowNet/lp_gflownet/gflownet/data/tree_dict.json.gz','r') as f_in:
    tree_dict=json.load(f_in)

class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        device="cpu",
        float_precision=32,
        env_id=None,
        #reward转换相关参数，energy相关的后续可以删除，reward_func可以再定义
        reward_beta=1,
        reward_norm=1.0,
        reward_norm_std_mult=0,
        reward_func="power",
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
        self.state = []#初始state为列表形式
        self.done = False
        self.n_actions = 0#不可舍弃，用于记录state在一条轨迹中的id（轨迹中的第几个state）
        self.id = env_id#用于计算轨迹id
        self.min_reward = 1e-8
        self.reward_beta = reward_beta
        self.reward_norm = reward_norm
        self.reward_norm_std_mult = reward_norm_std_mult
        self.reward_func = reward_func
        self.energies_stats = energies_stats
        self.denorm_proxy = denorm_proxy
        if oracle is None:#对gfn来说，只有proxy无oracle，所以self.oracle=proxy
            self.oracle = proxy
        else:
            self.oracle = oracle
        self._true_density = None
        self._z = None
        #设置两种动作空间（动作1和动作2），每个空间都要有终止动作
        self.action_space_1 = []
        self.action_space_2=[]
        self.eos_1 = len(self.action_space_1)
        self.eos_2 = len(self.action_space_2)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Assertions
        assert self.reward_norm > 0
        assert self.reward_beta > 0
        assert self.min_reward > 0
        #policy
        self.tree = HuffmanTree(count)#基于砌块的dict（砌块索引:砌块参与反应模板数）构建Huffman树
        self.tree_dict = tree_dict
                
    def copy(self):
        # return an instance of the environment
        return self.__class__(**self.__dict__)

    #设置reward_func相关
    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    #设置reward_func相关
    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm

    @abstractmethod
    def get_actions_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        pass

    #将policy_out_1和policy_out_2转换为np向量形式（全1）
    def get_fixed_policy_output_1(self):
        return np.ones(len(self.action_space_1))
    
    def get_fixed_policy_output_2(self):
        return np.ones(len(self.action_space_2))
    
    #返回最大轨迹长度(1e3)
    def get_max_traj_len(
        self,
    ):
        return 1e3

    #state(list)2proxy(product_smiles)，return self.statebatch2proxy([state])
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

    #将states(list[list])二维列表转换为np数组
    def statebatch2proxy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Prepares a batch of states in "GFlowNet format" for the proxy.
        """
        return np.array(states)

    #将states（张量形式,batch_size*state_dim)转换为proxy形式（张量，batch_size*state_proxy_dim)
    def statetorch2proxy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the proxy.输入的是smiles字符串形式，所以state_proxy_dim大小=smiles的字符串长度
        """
        return states

    #state2oracle的都可以不管，对env而言，只有proxy
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

    #计算单个state（包含done)的reward
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

    #返回一批states中已完成state（done=True）的reward列表
    def reward_batch(self, states: List[List], done=None):
        # Deprecated
        """
        Computes the rewards of a batch of states, given a list of states and 'dones'
        """
        if done is None:
            done = np.ones(len(states), dtype=bool)
        states_proxy = self.statebatch2proxy(states)[list(done), :]#从states中选取所有为True的行（这些行的所有列都会取出）
        rewards = np.zeros(len(done))
        if states_proxy.shape[0] > 0:
            rewards[list(done)] = self.proxy2reward(self.proxy(states_proxy)).tolist()#将rewards中已完成state的值变更为实际reward(未完成的则为0)
        return rewards

    #也是计算一批states的reward，但是输入的是张量形式，输出也是张量形式
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

    #从proxy值转换为reward数值（仅针对非主动学习情况）
    def proxy2reward(self, proxy_vals):
        """
        仅针对非主动学习的情况，将单个proxy的值（结合自由能，负值）转换为reward
        目前有四种可选的reward_func，不同func需要的预置参数不同
        """
        if self.denorm_proxy:
            proxy_vals = proxy_vals * (self.energies_stats[1] - self.energies_stats[0]) + self.energies_stats[0]
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

    #从reward数值转换回proxy值（仅针对非主动学习情况）
    #todo：需要check
    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into a (negative) energy or values as returned by
        an oracle.
        """
        if self.reward_func == "power":
            return torch.exp(
                (torch.log(reward) + self.reward_beta * np.log(self.reward_norm)- self.reward_beta * np.log(self.proxy_factor))
                / self.reward_beta
            )
        elif self.reward_func == "boltzmann":
            return torch.log(reward) / (self.reward_beta * self.proxy_factor)  
        elif self.reward_func == "identity":
            return  reward / self.proxy_factor
        elif self.reward_func == "linear_shift":
            return  (reward - self.reward_beta) / self.proxy_factor
        else:
            raise NotImplemented

    def state2policy_1(self, state=None):
        """
        Converts a state into a format suitable for a machine learning model, such as a
        one-hot encoding.
        """
        if state is None:
            state = self.state
        return state
    
    def state2policy_2(self, state=None):
        """
        Converts a state into a format suitable for a machine learning model, such as a
        one-hot encoding.
        """
        if state is None:
            state = self.state
        return state

    def statebatch2policy_1(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Converts a batch of states into a format suitable for a machine learning model,
        such as a one-hot encoding. Returns a numpy array.
        """
        return np.array(states)
    
    def statebatch2policy_2(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Converts a batch of states into a format suitable for a machine learning model,
        such as a one-hot encoding. Returns a numpy array.
        """
        return np.array(states)

    def statetorch2policy_1(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the policy
        """
        return states
    
    def statetorch2policy_2(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the policy
        """
        return states

    #以上区间为state2policy

    #舍弃
    def policy2state(self, state_policy: List) -> List:
        """
        Converts the model (e.g. one-hot encoding) version of a state given as
        argument into a state.
        """
        return state_policy

    #todo：state2readable的需要在sequence中修改，其作用是将state转换为product_smiles形式
    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state
        return str(state)

    #舍弃
    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        return readable
    
    #todo：需要修改（在sequence中)，理想的可读轨迹形式为砌块1smiles，砌块2smiles，反应1smarts，砌块3smiles，反应2smarts，产物smiles（即合成路线的形式呈现），如果是在mf里，再添加一个精度
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
    
    def get_parents(self, state=None, done=None):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : tuple
            Last action performed这个action可以从输入中删除

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
            return [state], [self.eos_1],[self.eos_2]
        else:
            parents = []
            actions_1 = []
            actions_2 = []
        return parents, actions_1,actions_2

    #基于policy_1的outputs采样一批actions_1（选择反应模板）
    def sample_actions_1(
        self,
        policy_outputs_1: TensorType["n_states", "policy_output_dim_1"],
        sampling_method: str = "policy_1",
        mask_invalid_actions_1: TensorType["n_states", "policy_output_dim_1"] = None,
        temperature_logits: float = 1.0,
        loginf: float = 1e6,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        device = policy_outputs_1.device
        ns_range = torch.arange(policy_outputs_1.shape[0]).to(device)
        if sampling_method == "uniform":
            logits = torch.ones(policy_outputs_1.shape).to(device)
        elif sampling_method == "policy":
            logits = policy_outputs_1
            logits /= temperature_logits
        if mask_invalid_actions_1 is not None:
            logits[mask_invalid_actions_1] = -loginf#mask_invalid_actions为True的位置设置为-loginf
        action_indices = Categorical(logits=logits).sample()#根据logits采样，采样获取的是某个logit对应的index,Categorical会自动将logits转换为概率分布
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]#根据logits计算logprobs
        # Build actions
        actions = [self.action_space_1[idx] for idx in action_indices]#根据action_indices获取actions
        return actions, logprobs   

    #list为policy_2的输出列表，mask_2为key
    def transform(self,list,mask_2):
        #基于mask_2这个key找到self.tree_dict对应的value（列表）
        #基于value列表，将其中的0/1的部分对list的相应位置进行修改
        #返回修改后的list
        value=self.tree_dict[mask_2]
        for i in range(len(value)):
            if value[i]==0:
                list[i]=0
            elif value[i]==1:
                list[i]=1
        return list
        
    #基于policy_2的outputs采样一批actions_2（选择反应砌块）
    #mask_2不再是列表，而作为一个字符串key
    def sample_actions_2(
        self,
        policy_outputs_2: TensorType["n_states", "policy_output_dim_2"],
        sampling_method: str = "policy_2",
        mask_invalid_actions_2: TensorType["n_states"] = None,
        temperature_logits: float = 1.0,
        loginf: float = 1e6,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        device = policy_outputs_2.device
        ns_range = torch.arange(policy_outputs_2.shape[0]).to(device)
        if sampling_method == "uniform":
            logits = torch.ones(policy_outputs_2.shape).to(device)
            action_indices = Categorical(logits=logits).sample()
            logprobs = self.logsoftmax(logits)[ns_range, action_indices]
            actions = [self.action_space_2[idx] for idx in action_indices]
            return actions, logprobs
        elif sampling_method == "policy_2":
            policy_outputs_2 = policy_outputs_2.to(self.device).detach().numpy().tolist()
            mask_invalid_actions_2 = mask_invalid_actions_2.to(self.device).detach().numpy().tolist()
            policy_outputs_2 = [self.transform(policy_outputs_2[i], mask_invalid_actions_2[i]) for i in range(policy_outputs_2.shape[0])]
            action_indices = [self.tree.find(p) for p in policy_outputs_2]#int
            pos_ret=[]
            neg_ret=[]
            for p in policy_outputs_2:
                x = self.tree.find(p)
                pos_nodes, neg_nodes = self.tree.get(x)   
                pos_current = [1 if i in pos_nodes else 0 for i in range(
                    len(self.tree))]
                neg_current = [1 if i in neg_nodes else 0 for i in range(
                    len(self.tree))]
                pos_ret.append(pos_current)
                neg_ret.append(neg_current)
            pos_ret=torch.LongTensor(pos_ret)
            neg_ret=torch.LongTensor(neg_ret)
            ones = torch.ones([ns_range, len(self.tree)])
            logprobs = torch.mul(pos_ret, torch.log(
                    policy_outputs_2)) + torch.mul(neg_ret, torch.log(torch.sub(ones, policy_outputs_2)))#计算输入分子最终对应的label的对数概率
            logprobs = torch.sum(logprobs, dim=1)#获得batch_size个分子的采样动作的对数概率
            actions = [self.action_space_2[idx] for idx in action_indices]#根据action_indices获取actions
            return actions, logprobs
 
    #todo:如果是backward,logprob为0，不需要输入mask
    def get_logprobs_1(
        self,
        policy_outputs_1: TensorType["n_states", "policy_output_dim_1"],
        is_forward: bool,
        actions_1: TensorType["n_states", 2],
        states_target: TensorType["n_states", "policy_input_dim_1"],
        mask_invalid_actions_1: TensorType["batch_size", "policy_output_dim_1"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        if is_forward:
            device = policy_outputs_1.device
            ns_range = torch.arange(policy_outputs_1.shape[0]).to(device)
            logits = policy_outputs_1
            if mask_invalid_actions_1 is not None:
                logits[mask_invalid_actions_1] = -loginf
            action_indices = (
            torch.tensor(
            [self.action_space_1.index(tuple(action.tolist())) for action in actions_1]
            )
            .to(int)
            .to(device)
            )
            logprobs = self.logsoftmax(logits)[ns_range, action_indices]
            return logprobs
        else:
            return torch.zeros(policy_outputs_1.shape[0], dtype=self.float, device=self.device)
    
    #actions_2不需要了，根据policy_outputs_2可以直接确认
    def get_logprobs_2(
        self,
        policy_outputs_2: TensorType["n_states", "policy_output_dim_2"],
        is_forward: bool,
        states_target: TensorType["n_states", "policy_input_dim_1"],
        mask_invalid_actions_2: TensorType["batch_size"] = None,
    ) -> TensorType["batch_size"]:
        if is_forward:
            device = policy_outputs_2.device
            ns_range = torch.arange(policy_outputs_2.shape[0]).to(device)
            policy_outputs_2 = policy_outputs_2.to(self.device).detach().numpy().tolist()
            mask_invalid_actions_2 = mask_invalid_actions_2.to(self.device).detach().numpy().tolist()
            policy_outputs_2 = [self.transform(policy_outputs_2[i], mask_invalid_actions_2[i]) for i in range(policy_outputs_2.shape[0])]
            pos_ret=[]
            neg_ret=[]
            for p in policy_outputs_2:
                x = self.tree.find(p)
                pos_nodes, neg_nodes = self.tree.get(x)   
                pos_current = [1 if i in pos_nodes else 0 for i in range(
                    len(self.tree))]
                neg_current = [1 if i in neg_nodes else 0 for i in range(
                    len(self.tree))]
                pos_ret.append(pos_current)
                neg_ret.append(neg_current)
            pos_ret=torch.LongTensor(pos_ret)
            neg_ret=torch.LongTensor(neg_ret)
            ones = torch.ones([ns_range, len(self.tree)])
            logprobs = torch.mul(pos_ret, torch.log(
                    policy_outputs_2)) + torch.mul(neg_ret, torch.log(torch.sub(ones, policy_outputs_2)))#计算输入分子最终对应的label的对数似然
            logprobs = torch.sum(logprobs, dim=1)#获得batch_size个分子的最终对数概率
            return logprobs
        else:
            return torch.zeros(policy_outputs_2.shape[0], dtype=self.float, device=self.device)

    def get_trajectories(
        self, traj_list, traj_actions_list_1, traj_actions_list_2,current_traj, current_actions_1,current_actions_2
    ):
        """
        Determines all trajectories leading to each state in traj_list, recursively.

        Args
        ----
        traj_list : list
            List of trajectories (lists)#每个轨迹都是一个list[list]形式

        traj_actions_list : list
            List of actions within each trajectory每个轨迹实际选用action的list形式

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
        parents, parents_actions_1,parents_actions_2 = self.get_parents(current_traj[-1], False)#当前轨迹的最后一个state的所有父节点及其action
        if parents == []:#重复递归，直到没有递归到没有父状态的初始状态时
            traj_list.append(current_traj)
            #对于actions_1而言，转换为元组形式，而actions_2不需要再考虑fid
            if hasattr(self, "action_pad_length"):
                # Required for compatibility with mfenv when length(sfenv_action) != length(fidelity.action)
                # For example, in AMP, length(sfenv_action) = 1 like (2,), length(fidelity.action) = 2 like (22, 1)
                current_actions_1 = [
                    tuple(list(action) + [0] * (self.action_max_length - len(action)))
                    for action in current_actions_1
                ]
            traj_actions_list_1.append(current_actions_1)
            traj_actions_list_2.append(current_actions_2)
            return traj_list, traj_actions_list_1,traj_actions_list_2
        for idx, (p, a1, a2) in enumerate(zip(parents, parents_actions_1,parents_actions_2)):
            traj_list, traj_actions_list_1,traj_actions_list_2 = self.get_trajectories(
                traj_list, traj_actions_list_1,traj_actions_list_2, current_traj + [p], current_actions_1 + [a1], current_actions_2+[a2]
            )
        return traj_list, traj_actions_list_1,traj_actions_list_2

    #具体细节在sequence中修改
    def step(self, action_idx_1,action_idx_2):
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
        if action_idx_1 < self.eos_1 and action_idx_2 < self.eos_2:
            self.done = False
            valid = True
        else:
            self.done = True
            valid = True
            self.n_actions += 1
        return self.state, action_idx_1,action_idx_2,valid

    #舍弃
    def no_eos_mask(self, state=None):
        """
        Returns True if no eos action is allowed given state
        """
        if state is None:
            state = self.state
        return False

    #sequence中修改
    def get_mask_invalid_actions_forward_1(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if forward action is
        invalid given the current state, False otherwise.
        #
        """
        mask = [False for _ in range(len(self.action_space_1))]
        return mask
    
    #sequence中修改
    def get_mask_invalid_actions_forward_2(self, state=None, done=None):
        mask = [False for _ in range(len(self.action_space_2))]
        return mask

    def get_mask_invalid_actions_backward_1(self, state=None, done=None, parents_a_1=None):
        """
        Returns a vector with the length of the discrete part of the action space + 1:
        True if action is invalid going backward given the current state, False
        otherwise.
        """
        if parents_a_1 is None:
            _, parents_a_1, parents_a_2 = self.get_parents()
        mask = [True for _ in range(len(self.action_space_1))]
        for pa in parents_a_1:
            mask[self.action_space_1.index(pa)] = False
        return mask
    
    #todo：由于动作2固定只有一种，是否可以不用mask（mask_b_1同理）
    def get_mask_invalid_actions_backward_2(self, state=None, done=None, parents_a_2=None):
        if parents_a_2 is None:
            _, parents_a_1, parents_a_2 = self.get_parents()
        mask = [True for _ in range(len(self.action_space_1))]
        for pa in parents_a_2:
            mask[self.action_space_2.index(pa)] = False
        return mask
    
    #更新state
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
    #Buffer的作用
    1.训练样本的记录缓冲区、训练数据集和测试数据集的缓冲区
    2.对各种数据缓存区进行一些基本操作，例如插入新的数据、删除旧的数据、从缓存区中随机采样一批数据
    """

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
        self.replay.reward = pd.to_numeric(self.replay.reward)#将replay.reward转换为数值类型
        self.replay.energy = pd.to_numeric(self.replay.energy)
        self.replay.reward = [-1 for _ in range(self.replay_capacity)]#将replay.reward初始化为-1
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
            env.buffer.train.output_pkl.#离线轨迹将不会被采样。为了采样离线轨迹，缓冲区的训练配置应该是完整的和可行的，并且在env.buffer.train.output_pkl中应定义一个输出pkl文件。
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
            env.buffer.test.output_pkl.#测试指标将不会被计算。为了计算测试指标，缓冲区的测试配置应该是完整的和可行的，并且在env.buffer.test.output_pkl中应定义一个输出pkl文件。
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

    #buffer选择有两种，一种是"main",一种是"replay"
    #输入states, trajs, rewards, energies, it,将轨迹数据存储到df中（main或replay中）
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
                            "state": [self.env.state2readable(s) for s in states],#[产物smiles1,产物smiles2,产物smiles3,...]
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
                self.replay = self._add_greater(states, trajs, rewards, energies, it)

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
        while np.max(rewards_new) > np.min(rewards_old):
            idx_new_max = np.argmax(rewards_new)
            readable_state = self.env.state2readable(states[idx_new_max])
            if not self.replay["state"].isin([readable_state]).any():
                self.replay.iloc[self.replay.reward.argmin()] = {
                    "state": self.env.state2readable(states[idx_new_max]),
                    "traj": self.env.traj2readable(trajs[idx_new_max]),
                    "reward": rewards[idx_new_max],
                    "energy": energies[idx_new_max],
                    "iter": it,
                }
                rewards_old = self.replay["reward"].values
            rewards_new[idx_new_max] = -1
        return self.replay

    def make_data_set(self, config):
        """
        Constructs a data set as a DataFrame according to the configuration.
        """
        stats = None
        if config is None:
            return None, None, None
        elif "path" in config and config.path is not None:
            path = self.logger.logdir / Path("data") / config.path
            df = pd.read_csv(path, index_col=0)
            samples = [self.env.readable2state(s) for s in df["samples"].values]
            stats = self.compute_stats(df)
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
