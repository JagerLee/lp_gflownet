import sys
import copy
import time
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import pickle#pickle后缀的文件，用于保存和提取
from torch.distributions import Categorical, Bernoulli#概率分布
from tqdm import tqdm
from scipy.special import logsumexp#logsumexp函数
import random
from gflownet.envs.base import Buffer
from gflownet.utils.common import set_device, set_float_precision, torch2np
from gflownet.utils.huffman import HuffmanTree
from torchtyping import TensorType
import json
import gzip
with gzip.GzipFile('/home/lishuwang/project/GFlowNet/lp_gflownet/gflownet/data/smiles_list.json.gz','r') as f_in:
    smiles_list = json.load(f_in)

#states:[state1,state2,..]，每个state通过step进行更新（更新为state_new）,每个state相当于一条轨迹，states相当于一批轨迹
class GFlowNetAgent:
    def __init__(
        self,
        env,
        seed,
        device,
        float_precision,
        optimizer,
        buffer,
        policy,
        mask_invalid_actions,
        temperature_logits,
        random_action_prob,
        pct_offline,
        logger,
        num_empirical_loss,
        oracle,
        proxy=None,
        active_learning=False,
        data_path=None,
        sample_only=False,
        **kwargs,
    ):
        # Seed
        self.rng = np.random.default_rng(seed)#生成特定分布（正态/泊松）下的随机数。种子数的提供可以保证每次生成的随机数相同
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Environment
        self.env = env
        self.mask_source_1 = self._tbool([self.env.get_mask_invalid_actions_forward_1()])#这里的_tbool是将bool类型的数组转换为tensor类型；
        # self.mask_source_2 = self._tbool([self.env.get_mask_invalid_actions_forward_2()])
        
        # 基于输入的参数optimizer确定选择的损失函数类型
        if optimizer.loss in ["flowmatch", "flowmatching"]:
            self.loss = "flowmatch"
            self.logZ = None
        elif optimizer.loss in ["trajectorybalance", "tb"]:
            self.loss = "trajectorybalance"
            self.logZ = nn.Parameter(torch.ones(optimizer.z_dim) * 150.0 / 64)#模型参数：logZ，其张量的形状为(optimizer.z_dim),初始值为1*150/64
            #由于被封装在nn.Parameter()函数中，所以logZ可以被优化器在训练过程进行值更新
        else:
            print("Unkown loss. Using flowmatch as default")
            self.loss = "flowmatch"
            self.logZ = None
            
        # loss_eps is used only for the flowmatch loss该参数仅用于flowmatch损失
        self.loss_eps = torch.tensor(float(1e-5)).to(self.device)
        
        # Logging#日志记录
        self.num_empirical_loss = num_empirical_loss#计算神经网络损失函数的经验样本数量
        self.logger = logger#用于记录日志
        self.oracle_n = oracle.n#表示oracle的训练数据大小
        
        # Buffers用于存储数据（训练集，测试集）和提供数据分析
        self.buffer = Buffer(
            **buffer, env=self.env, make_train_test=not sample_only, logger=logger
        )#sample_only决定是否只采样，不进行训练和测试集的获取
        
        # Train set statistics and reward normalization constant 训练集数据分析和奖励归一化常数
        if self.buffer.train is not None:
            energies_stats_tr = [
                self.buffer.min_tr,
                self.buffer.max_tr,
                self.buffer.mean_tr,
                self.buffer.std_tr,
                self.buffer.max_norm_tr,
            ]
            self.env.set_energies_stats(energies_stats_tr)
            print("\nGFN Train data")
            print(f"\tMean score: {energies_stats_tr[2]}")
            print(f"\tStd score: {energies_stats_tr[3]}")
            print(f"\tMin score: {energies_stats_tr[0]}")
            print(f"\tMax score: {energies_stats_tr[1]}")
        else:
            energies_stats_tr = None
        ##上述为训练集的统计信息    
        
        if self.env.reward_norm_std_mult > 0 and energies_stats_tr is not None:
            self.env.reward_norm = self.env.reward_norm_std_mult * energies_stats_tr[3]
            self.env.set_reward_norm(self.env.reward_norm)
        #当使用power作为reward function，reward_norm=reward_norm_std_mult*std(energies)，即能量标准差的reward_norm_std_mult倍数
        
        # Test set statistics测试集数据分析
        if self.buffer.test is not None:
            print("\nGFN Test Data")
            print(f"\tMean score: {self.buffer.mean_tt}")
            print(f"\tStd score: {self.buffer.std_tt}")
            print(f"\tMin score: {self.buffer.min_tt}")
            print(f"\tMax score: {self.buffer.max_tt}")
            
            
        
        #前向policy_model_1和policy_model_2设置
        #policy.forward在config中
        self.forward_policy_1 = Policy(policy.forward_1, self.env, self.device, self.float)
        #todo:set_forward_policy_ckpt_path修改
        if "checkpoint" in policy.forward_1 and policy.forward_1.checkpoint:
            self.logger.set_forward_policy_ckpt_path(policy.forward_1.checkpoint)
            if False:
                self.forward_policy.load_state_dict(
                    torch.load(self.policy_forward_path)
                )
                print("Reloaded GFN forward policy model Checkpoint")
        else:
            self.logger.set_forward_policy_ckpt_path(None)
        
        with gzip.GzipFile(policy.count_path, 'r') as f_in:
            count = json.load(f_in)
        self.forward_policy_2 = Policy(
            config=policy.forward_2,
            env=self.env,
            device=self.device,
            float_precision=self.float,
            count=count
        )
        #todo:set_forward_policy_ckpt_path修改
        if "checkpoint" in policy.forward_2 and policy.forward_2.checkpoint:
            self.logger.set_forward_policy_ckpt_path(policy.forward_2.checkpoint)
            if False:
                self.forward_policy.load_state_dict(
                    torch.load(self.policy_forward_path)
                )
                print("Reloaded GFN forward policy model Checkpoint")
        else:
            self.logger.set_forward_policy_ckpt_path(None)   
            
            
            
            
            
        #后向policy_model_1和policy_model_2设置
        self.backward_policy_1 = Policy(
            policy.backward_1,
            self.env,
            self.device,
            self.float,
            base=self.forward_policy_1,
        )
        if (
            policy.backward_1
            and "checkpoint" in policy.backward_1
            and policy.backward_1.checkpoint
        ):
            self.logger.set_backward_policy_ckpt_path(policy.backward_1.checkpoint)
            if False:
                self.backward_policy.load_state_dict(
                    torch.load(self.policy_backward_path)
                )
                print("Reloaded GFN backward policy model Checkpoint")
        else:
            self.logger.set_backward_policy_ckpt_path(None)
        self.ckpt_period = policy.ckpt_period
        if self.ckpt_period in [None, -1]:
            self.ckpt_period = np.inf
        
        self.backward_policy_2 = Policy(
            policy.backward_2,
            self.env,
            self.device,
            self.float,
            base=self.forward_policy_2,
        )
        if (
            policy.backward_2
            and "checkpoint" in policy.backward_2
            and policy.backward_2.checkpoint
        ):
            self.logger.set_backward_policy_ckpt_path(policy.backward_2.checkpoint)
            if False:
                self.backward_policy.load_state_dict(
                    torch.load(self.policy_backward_path)
                )
                print("Reloaded GFN backward policy model Checkpoint")
        else:
            self.logger.set_backward_policy_ckpt_path(None)
        self.ckpt_period = policy.ckpt_period
        if self.ckpt_period in [None, -1]:
            self.ckpt_period = np.inf
        
        
        if self.forward_policy_1.is_model:#如果forward_policy是模型（采用TB为目标函数）
            self.target_1 = copy.deepcopy(self.forward_policy_1.model)#深拷贝模型
            self.opt_1, self.lr_scheduler_1 = make_opt(
                self.parameters_1(), self.logZ, optimizer
            )
        else:
            self.opt_1, self.lr_scheduler_1, self.target_1 = None, None, None
        
        if self.forward_policy_2.is_model:#如果forward_policy是模型（采用TB为目标函数）
            self.target_2 = copy.deepcopy(self.forward_policy_2.model)#深拷贝模型
            self.opt, self.lr_scheduler_2 = make_opt(
                self.parameters_2(), self.logZ, optimizer
            )
        else:
            self.opt_2, self.lr_scheduler_2, self.target_2 = None, None, None
        
        
        self.n_train_steps = optimizer.n_train_steps#训练步数
        self.batch_size = optimizer.batch_size
        self.ttsr = max(int(optimizer.train_to_sample_ratio), 1)#train to sample ratio,训练步骤与数据采样之间的比率（经过多少个训练步骤才从数据集中采样一次并更新模型参数）
        self.sttr = max(int(1 / optimizer.train_to_sample_ratio), 1)#采样一次需要多少个训练步骤
        self.clip_grad_norm = optimizer.clip_grad_norm
        self.tau = optimizer.bootstrap_tau#用于计算logits的温度参数
        self.ema_alpha = optimizer.ema_alpha#指数移动平均的参数
        self.early_stopping = optimizer.early_stopping
        self.use_context = active_learning
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)#对dim=1的维度进行logsoftmax操作
        # Training
        self.mask_invalid_actions = mask_invalid_actions#True/False
        self.temperature_logits = temperature_logits
        self.random_action_prob = random_action_prob
        self.pct_offline = pct_offline#用于计算经验样本的比例
        # Metrics
        self.l1 = -1.0
        self.kl = -1.0
        self.jsd = -1.0

    #2tensor,float
    def _tfloat(self, x):
        if isinstance(x, torch.Tensor):  # logq
            return x.to(self.device, self.float)
        elif isinstance(x[0], torch.Tensor):  # state is already a tensor
            x = [x_element.unsqueeze(0) for x_element in x]
            return torch.cat(x).to(self.device, self.float)
        else:  # if x is a list
            return torch.tensor(x, dtype=self.float, device=self.device)

    #2tensor,long
    def _tlong(self, x):
        return torch.tensor(x, dtype=torch.long, device=self.device)#将x转换为tensor类型，数据类型为long

    #2tensor,int
    def _tint(self, x):
        return torch.tensor(x, dtype=torch.int, device=self.device)

    #2tensor,bool
    def _tbool(self, x):
        return torch.tensor(x, dtype=torch.bool, device=self.device)

    #return list(self.forward_policy.model.parameters()) + list(self.backward_policy.model.parameters()
    def parameters_1(self):
        if self.backward_policy_1.is_model == False:
            return list(self.forward_policy_1.model.parameters())
        elif self.loss == "trajectorybalance":
            return list(self.forward_policy_1.model.parameters()) + list(
                self.backward_policy_1.model.parameters()
            )
        else:
            raise ValueError("Backward Policy cannot be a nn in flowmatch.")
    def parameters_2(self):
        if self.backward_policy_2.is_model == False:
            return list(self.forward_policy_2.model.parameters())
        elif self.loss == "trajectorybalance":
            return list(self.forward_policy_2.model.parameters()) + list(
                self.backward_policy_2.model.parameters()
            )
        else:
            raise ValueError("Backward Policy cannot be a nn in flowmatch.")

    #sampling_method="policy",返回每个state的动作（基于env.sample_actions）
    #动作1，反应模板选择
    def sample_actions_1(
        self,
        envs,
        times,
        sampling_method="policy_1",
        model=None,#forward_policy/backward_policy
        is_forward: bool = True,
        temperature=1.0,
        random_action_prob=0.0,#每个state采用随机动作而非policy动作的概率（具体实现看idx_norandom）
    ):
        """
        #输入的是一批states[state1,state2,state3...]
        state1:[index1,index2,index3...]
        
        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment
            
        times : dict
            Dictionary to store times

        sampling_method : str   采样方法（用policy model还是从动作空间中均匀采样）
            - model: uses current forward to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space(每个动作的logits相等)

        model : policy model
            Model to use as policy if sampling_method="policy"

        is_forward : bool
            True if sampling is forward. False if backward.

        temperature : float#用于调整policy的输出值
            Temperature to adjust the logits by logits /= temperature
            
        """
        
        # backward_sample.反向采样的设定
        if sampling_method == "random":
            random_action_prob = 1.0#每个state采用随机采样的概率变为100%
        if not isinstance(envs, list):
            envs = [envs]
        # Build states and masks
        states = [env.state for env in envs]#states:[state1,state2,state3...]
        
        #mask_invalid_actions（前向/反向）：对于envs中的每个env(即state)，返回每个state的mask列表（动作空间，无效动作为True）的列表
        if is_forward:
            mask_invalid_actions_1 = self._tbool(
                [env.get_mask_invalid_actions_forward_1() for env in envs]
            )
        else:
            mask_invalid_actions_1 = self._tbool(
                [env.get_mask_invalid_actions_backward_1() for env in envs]
            )
        
        #policy_outputs：每个state返回所有动作空间的动作logits值

        #todo:当前（原gfn）思路的采样是随机和策略采样的混合，考虑到当前任务采样的步骤较少，可以全部采用策略采样
        policy_outputs = model.random_distribution(states)#每个state的所有action的logit为1
        
        #idx_norandom：[True,False,True...],每个bool对应一个state，True表示当前state不采用随机动作
        idx_norandom = (
            Bernoulli(
                (1 - random_action_prob) * torch.ones(len(states), device=self.device)
            )#Bernoulli分布是二项分布（只有0，1），sample的只会是0或1（采样的概率为1-random_action_prob，如果random_action_prob为0，则idx_norandom都为True）
            .sample()
            .to(bool)
        )
        
        # Check for at least one non-random action
        if sampling_method == "policy_1":
            if idx_norandom.sum() > 0:#统计idx_norandom中有几个True(即非随机动作)
                policy_outputs[idx_norandom, :] = model(
                    self._tfloat(
                        self.env.statebatch2policy_1(
                            [s for s, do in zip(states, idx_norandom) if do]
                        )
                    )
                )#policy_outputs[idx_norandom, :]表示从policy_outputs数组中选取行索引为idx_norandom中值为True的所有行，然后选取这些行的所有列
        else:
            raise NotImplementedError
        
        #基于policy的输出logits，返回每个state最终采取的动作
        #这里的logprobs是每个实际采样动作的logits的log值
        actions, logprobs = self.env.sample_actions_1(
            policy_outputs,
            sampling_method,
            mask_invalid_actions_1,
            temperature,
        )
        return actions
    
    #动作2，反应砌块选择，这里删除了随机采样，仅考虑策略采样
    def sample_actions_2(
        self,
        envs,
        times,
        sampling_method="policy_2",
        model=None,#forward_policy/backward_policy
        is_forward: bool = True,
        temperature=1.0,
        #random_action_prob=0.0,#每个state采用随机动作而非policy动作的概率（具体实现看idx_norandom）
        actions_1=None,
    ):
        """
        #输入的是一批states[state1,state2,state3...]
        state1:[index1,index2,index3...]
        
        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment
            
        times : dict
            Dictionary to store times

        sampling_method : str   采样方法（用policy model还是从动作空间中均匀采样）
            - model: uses current forward to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space(每个动作的logits相等)

        model : policy model
            Model to use as policy if sampling_method="policy"

        is_forward : bool
            True if sampling is forward. False if backward.

        temperature : float#用于调整policy的输出值
            Temperature to adjust the logits by logits /= temperature
            
        """
        if not isinstance(envs, list):
            envs = [envs]
        # Build states and masks
        states = [env.state for env in envs]#states:[state1,state2,state3...]
        
        #mask_invalid_actions（前向/反向）：对于envs中的每个env(即state)，返回每个state的mask列表（动作空间，无效动作为True）的列表
        if is_forward:
            mask_invalid_actions_2 = self._tbool(
                [env.get_mask_invalid_actions_forward_2() for env in envs]
            )
        else:
            mask_invalid_actions_2 = self._tbool(
                [env.get_mask_invalid_actions_backward_2() for env in envs]
            )
        
        #policy_outputs：每个state返回所有动作空间的动作logits值

        policy_outputs = model(self._tfloat(self.env.statebatch2policy_2(states,actions_1)))
        
        #基于policy的输出logits，返回每个state最终采取的动作
        actions, logprobs = self.env.sample_actions_2(
            policy_outputs,
            sampling_method,
            mask_invalid_actions_2,
            temperature,
        )
        return actions

    #todo:是否需要设定step仅限于前三步
    def step(
        self,
        envs: List,
        actions_1: List[Tuple],
        actions_2: List[Tuple],
        is_forward: bool = True,
    ):
        """
        Executes the actions of a list on the environments of a list.#执行动作列表中的动作

        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment

        actions : list
            A list of actions to be executed on each env of envs.

        is_forward : bool
            True if sampling is forward. False if backward.

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """
        assert len(envs) == len(actions_1)#每个state一个action
        assert len(envs) == len(actions_2)
        
        if not isinstance(envs, list):
            envs = [envs]
        
        if is_forward:#如果是前向传播
            envs, actions_1,actions_2, valids = zip(
                *[env.step(action_1,action_2) for env, action_1,action_2 in zip(envs, actions_1,actions_2)]
            )#对于每个env（state），执行对应的动作，返回下一个state与是否有效。最终返回final_state，源动作列表，是否有效列表
        
        else:#获取每个state的前一个state，前一个action
            valids = []
            for env, action_1,action_2 in zip(envs, actions_1,actions_2):
                parents, parents_a_1,parents_a_2 = env.get_parents()
                if action_1 in parents_a_1 and action_2 in parents_a_2:
                    state_next = parents#因为当前设置下，每个state都只有一个父节点
                    env.n_actions -= 1
                    env.set_state(state_next, done=False)#更新envs中所有env(即state)
                    valids.append(True)
                else:
                    valids.append(False)
        return envs, actions_1, actions_2, valids

    #input:envs,n_samples（用于训练结束后采样）
    # 正向+反向（离线）采样获得一个batch(每个state都是有state id和traj id的),return batch+times
    #todo:env.id确认
    def sample_batch(
        self, envs, n_samples=None, train=True, model=None, progress=False
    ):
        """
        Builds a batch of data
        
        #batch:[state1,state2..]这里的batch会收录所有的state，包括离线样本（即反向传播）和正向样本的所有state
        如[state1,state2,state3..state1',state2',state3'...]
        
        if train == True:
            Each item in the batch is a list of 7 elements (all tensors):
                - [0] the state
                - [1] the action_1
                - [2] the action_2
                - [3] all parents of the state
                - [4] actions_1 that lead to the state from each parent
                - [5] actions_2 that lead to the state from each parent
                - [6] done [True, False]
                - [7] traj id: identifies each trajectory
                - [8] state id: identifies each state within a traj
                - [9] mask_f_1: invalid forward actions from that state are 1#无效动作用1表示
                - [10] mask_f_2
                - [11] mask_b_1: invalid backward actions from that state are 1
                - [12] mask_b_2
            修改: action改为action_1,action_2
            parent_a改为parent_a_1,parent_a_2
            mask_f改为mask_f_1,mask_f_2
        else:
            Each item in the batch is a list of 1 element:
                - [0] the states 

        Args
        ----
        """

        def _add_to_batch(batch, envs, actions_1, actions_2, valids, train=True):
            for env, action_1, action_2, valid in zip(envs, actions_1, actions_2, valids):
                if valid is False:#如果当前动作无效，则跳过该state在当前动作下的更新
                    continue
                parents, parents_a_1, parents_a_2 = env.get_parents()
                mask_f_1 = env.get_mask_invalid_actions_forward_1()
                mask_f_2 = env.get_mask_invalid_actions_forward_2()
                mask_b_1 = env.get_mask_invalid_actions_backward_1(
                    env.state, env.done, parents_a_1
                )
                mask_b_2 = env.get_mask_invalid_actions_backward_2(
                    env.state, env.done, parents_a_2
                )
                
                if train:
                    batch.append(
                        [
                            self._tfloat([env.state]),
                            self._tfloat([action_1]),
                            self._tfloat([action_2]),
                            self._tfloat(parents),
                            self._tfloat(parents_a_1),
                            self._tfloat(parents_a_2),
                            self._tbool([env.done]),
                            self._tlong([env.id] * len(parents)),
                            self._tlong([env.n_actions]),#state id:一个轨迹中的每个state的id（1，2，3）
                            self._tbool([mask_f_1]),
                            self._tbool([mask_f_2]),
                            self._tbool([mask_b_1]),
                            self._tbool([mask_b_2]),
                        ]
                    )

                else:#如果只采样，batch：[state1,state2,...]
                    if env.done:
                        batch.append(env.state)
            return batch

        times = {
            "all": 0.0,
            "forward_actions": 0.0,
            "backward_actions": 0.0,
            "actions_envs": 0.0,
            "rewards": 0.0,
        }#记录时间
        t0_all = time.time()#记录开始时间
        
        #构建batch
        batch = []
        
        #清洗envs顺序
        if isinstance(envs, list):
            envs = [env.reset(idx) for idx, env in enumerate(envs)]
        #前n_samples个样本进行reset（恢复为初始state）
        elif n_samples is not None and n_samples > 0:
            envs = [self.env.copy().reset(idx) for idx in range(n_samples)]
        else:
            return None, None
        
        # Offline trajectories
        if train:
            n_empirical = int(self.pct_offline * len(envs))#len(envs)样本数*pct_offline离线样本采样比例
        else:
            n_empirical = 0
            
        if n_empirical > 0 and self.buffer.train_pkl is not None:
            with open(self.buffer.train_pkl, "rb") as f:
                dict_tr = pickle.load(f)
                x_tr = dict_tr["x"]#state列
            random.shuffle(x_tr)
            envs_offline = []
            actions = []
            valids = []
            for idx in range(n_empirical):
                env = envs[idx]#初始state
                env.n_actions = env.get_max_traj_len()#每个轨迹的state索引范围
                # Required for env.done check in the mfenv env
                env = env.set_state(x_tr[idx], done=True)#导入指定索引的state
                envs_offline.append(env)#将导入的state添加到envs_offline中
                actions.append((env.eos,))
                valids.append(True)
        else:
            envs_offline = []
            
        # Sample backward actions
        #将所有经验样本的所有states加入到batch中（只采样，即只更新state）
        while envs_offline:
            with torch.no_grad():
                actions_1 = self.sample_actions_1(
                    envs_offline,
                    times,
                    sampling_method="policy_1",
                    model=self.backward_policy_1,
                    is_forward=False,
                    temperature=self.temperature_logits,
                    random_action_prob=self.random_action_prob,
                )
                actions_2 = self.sample_actions(
                    envs_offline,
                    times,
                    sampling_method="policy_2",
                    model=self.backward_policy_2,
                    is_forward=False,
                    temperature=self.temperature_logits,
                    random_action_prob=self.random_action_prob,
                )
            
            batch = _add_to_batch(batch, envs_offline, actions_1, actions_2, valids)
            # 基于反向传播采样的动作，往前更新envs_offline
            envs_offline, actions_1, actions_2, valids = self.step(
                envs_offline, actions_1, actions_2, is_forward=False
            )
            assert all(valids)
            
            # Filter out finished trajectories
            if isinstance(env.state, list):
                envs_offline = [env for env in envs_offline if env.state != env.source]#将已经返回为初始state的轨迹（反向传播完成）去除
            elif isinstance(env.state, TensorType):
                envs_offline = [
                    env
                    for env in envs_offline
                    if not torch.eq(env.state, env.source).all()
                ]

        envs = envs[n_empirical:]#去除离线样本
        
        # Sample forward actions
        # 正向轨迹获取
        while envs:
            with torch.no_grad():
                if train is False:
                    actions_1 = self.sample_actions_1(
                        envs,
                        times,
                        sampling_method="policy_1",
                        model=self.forward_policy_1,
                        is_forward=True,
                        temperature=1.0,
                        random_action_prob=self.random_action_prob,
                    )
                    actions_2 = self.sample_actions_2(
                        envs,
                        times,
                        sampling_method="policy_2",
                        model=self.forward_policy_2,
                        is_forward=True,
                        temperature=1.0,
                        random_action_prob=self.random_action_prob,
                    )
                else:
                    actions_1 = self.sample_actions_1(
                        envs,
                        times,
                        sampling_method="policy_1",
                        model=self.forward_policy_1,
                        is_forward=True,
                        temperature=self.temperature_logits,#如果是用于训练，需要调整policy的输出值
                        random_action_prob=self.random_action_prob,
                    )
                    actions_2 = self.sample_actions_2(
                        envs,
                        times,
                        sampling_method="policy_2",
                        model=self.forward_policy_2,
                        is_forward=True,
                        temperature=self.temperature_logits,#如果是用于训练，需要调整policy的输出值
                        random_action_prob=self.random_action_prob,
                    )
            # Update environments with sampled actions
            envs, actions_1, actions_2, valids = self.step(envs, actions_1,actions_2, is_forward=True)
            # Add to batch
            t0_a_envs = time.time()
            batch = _add_to_batch(batch, envs, actions_1, actions_2, valids, train)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]#去除采样已经完成的轨迹
            t1_a_envs = time.time()#记录结束时间
            times["actions_envs"] += t1_a_envs - t0_a_envs#记录时间
            if progress and n_samples is not None:
                print(f"{n_samples - len(envs)}/{n_samples} done")#n_samples为目标采样数，len(envs)为仍未采样完成的数目

        return batch, times
 
    
    def flowmatch_loss(self, it, batch, loginf=1000):
        
        """
        Computes the loss of a batch

        Args
        ----
        it : int
            Iteration

        batch : ndarray
            A batch of data: every row is a state (list), corresponding to all states
            visited in each state in the batch.

        Returns
        -------
        loss : float
            Loss, as per Equation 12 of https://arxiv.org/abs/2106.04399v1

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        loginf = self._tfloat([loginf])
        # Unpack batch
        (
            states,
            actions_1,
            actions_2,
            parents,
            parents_a_1,
            parents_a_2,
            done,
            traj_id_parents,
            state_id,
            masks_sf_1,#1代表无效动作
            masks_sf_2,
            masks_b_1,
            masks_b_2,
        ) = zip(*batch)
        # Get state/batch id
        parents_batch_id = self._tlong(
            sum([[idx] * len(3) for idx, p in enumerate(parents)], [])#将parents中的每个元素（列表）的索引号重复state长度次，然后将所有的列表合并
        )#eg:parents_batch_id = [0,0,0,1,1,1,2,2,2...]0为parent的索引，重复3次，即单个轨迹的有效state为3个
        
        states, parents, actions_1,actions_2, parents_a_1, parents_a_2, done, masks_sf_1, masks_sf_2 = map(
            torch.cat,
            [
                states,
                parents,
                actions_1,
                actions_2,
                parents_a_1,
                parents_a_2,
                done,
                masks_sf_1,
                masks_sf_2,
            ],
        )
        actions_1 = actions_1.to(int).squeeze()
        actions_2 = actions_2.to(int).squeeze()
        parents_a_1 = parents_a_1.to(int).squeeze()
        parents_a_2 = parents_a_2.to(int).squeeze()
        
        # Compute rewards
        rewards = self.env.reward_torchbatch(states, done)
        assert torch.all(rewards[done] > 0)#所有final state的reward都大于0
        
        # In-flows
        inflow_logits_1 = -loginf * torch.ones(
            (states.shape[0], self.env.policy_output_dim_1),
            device=self.device,
        )#初始化
        inflow_logits_2 = -loginf * torch.ones(
            (states.shape[0], self.env.policy_output_dim_2),
            device=self.device,
        )
        
        inflow_logits_1[parents_batch_id, parents_a_1] = self.forward_policy_1(
            self.env.statetorch2policy_1(parents)
        )[torch.arange(parents.shape[0]), parents_a_1]
        
        inflow_logits_2[parents_batch_id, parents_a_2] = self.forward_policy_2(
            self.env.statetorch2policy_2(parents,parents_a_1)
        )[torch.arange(parents.shape[0]), parents_a_2]
        
        
        inflow_1 = torch.logsumexp(inflow_logits_1, dim=1)#只有非负无穷值会被计入
        inflow_2 = torch.logsumexp(inflow_logits_2, dim=1)
        inflow = inflow_1 + inflow_2
        
        # Out-flows
        outflow_logits_1 = self.forward_policy_1(self.env.statetorch2policy_1(states))
        outflow_logits_2 = self.forward_policy_2(self.env.statetorch2policy_2(states,actions_1))
        
        outflow_logits_1[masks_sf_1] = -loginf
        outflow_logits_2[masks_sf_2] = -loginf
        
        outflow_1 = torch.logsumexp(outflow_logits_1, dim=1)
        outflow_2 = torch.logsumexp(outflow_logits_2, dim=1)
        
        outflow_1 = outflow_1 * torch.logical_not(done) - loginf * done
        outflow_2 = outflow_2 * torch.logical_not(done) - loginf * done
        
        outflow = torch.logaddexp(torch.log(rewards), outflow_1, outflow_2)
        
        # Flow matching loss
        loss = (inflow - outflow).pow(2).mean()
        # Isolate loss at terminating nodes and all other nodes
        with torch.no_grad():
            term_loss = ((inflow - outflow) * done).pow(2).sum() / (done.sum() + 1e-20)
            flow_loss = ((inflow - outflow) * torch.logical_not(done)).pow(2).sum() / (
                torch.logical_not(done).sum() + 1e-20
            )
        return (loss, term_loss, flow_loss), rewards[done.eq(1)]

    #todo:loss的计算公式需要再确认
    def trajectorybalance_loss(self, it, batch, loginf=1000):
        """
        Computes the trajectory balance loss of a batch

        Args
        ----
        it : int
            Iteration

        batch : ndarray
            A batch of data: every row is a state (list), corresponding to all states
            visited in each state in the batch.

        Returns
        -------
        loss : float

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        loginf = self._tfloat([loginf])
        # Unpack batch
        (
            states,
            actions_1,
            actions_2,
            parents,
            parents_a_1,
            parents_a_2,
            done,
            traj_id_parents,
            state_id,
            masks_sf_1,
            masks_sf_2,
            masks_b_1,
            masks_b_2,
        ) = zip(*batch)
        # Keep only parents in trajectory
        parents = [
            p[torch.logical_and(torch.all(torch.eq(a_1, p_a_1), dim=1), torch.all(torch.eq(a_2, p_a_2), dim=1))]
            for a_1, a_2, p, p_a_1, p_a_2 in zip(actions_1,actions_2, parents, parents_a_1, parents_a_2)
        ]
        traj_id = torch.cat([el[:1] for el in traj_id_parents])
        # Concatenate lists of tensors
        states, actions_1, actions_2, parents, done, state_id, masks_sf_1,masks_sf_2,masks_b_1, masks_b_2 = map(
            torch.cat,
            [
                states,
                actions_1,
                actions_2,
                parents,
                done,
                state_id,
                masks_sf_1,
                masks_sf_2,
                masks_b_1,
                masks_b_2,
            ],
        )
        
        # Shift state_id to [1, 2, ...]
        for tid in traj_id.unique():
            state_id[traj_id == tid] = (
                state_id[traj_id == tid] - state_id[traj_id == tid].min()
            ) + 1
            # state_id[traj_id == tid] -= state_id[traj_id == tid].min() + 1
            
        # Compute rewards
        rewards = self.env.reward_torchbatch(states, done)
        
        # Build parents forward masks from state masks
        masks_f_1 = torch.cat(
            [
                masks_sf_1[torch.where((state_id == sid - 1) & (traj_id == pid))]
                if sid > 1
                else self.mask_source
                for sid, pid in zip(state_id, traj_id)
            ]
        )#masks_f_1：每个state的前一个state的mask
        masks_f_2 = torch.cat(
            [
                masks_sf_2[torch.where((state_id == sid - 1) & (traj_id == pid))]
                if sid > 1
                else self.mask_source
                for sid, pid in zip(state_id, traj_id)
            ]
        )
        
        
        # Forward trajectories
        policy_output_f_1 = self.forward_policy_1(self.env.statetorch2policy_1(parents))
        policy_output_f_2 = self.forward_policy_2(self.env.statetorch2policy_2(parents,actions_1))
        
        logprobs_f_1 = self.env.get_logprobs_1(
            policy_output_f_1, True, actions_1, states, masks_f_1, loginf
        )#logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        
        logprobs_f_2 = self.env.get_logprobs_2(
            policy_output_f_2, True, actions_2, states, masks_f_2, loginf
        )
        
        sumlogprobs_f_1 = torch.zeros(
            len(torch.unique(traj_id, sorted=True)),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_id, logprobs_f_1)
        
        sumlogprobs_f_2 = torch.zeros(
            len(torch.unique(traj_id, sorted=True)),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_id, logprobs_f_2)
        
        
        # Backward trajectories
        policy_output_b_1 = self.backward_policy_1(self.env.statetorch2policy_1(states))
        policy_output_b_2 = self.backward_policy_2(self.env.statetorch2policy_2(states,actions_1))
        
        
        logprobs_b_1 = self.env.get_logprobs_1(
            policy_output_b_1, False, actions_1, parents, masks_b_1, loginf
        )#parents和masks_b_1可以不用
        
        logprobs_b_2 = self.env.get_logprobs_2(
            policy_output_b_2, False
        )
        
        sumlogprobs_b_1 = torch.zeros(
            len(torch.unique(traj_id, sorted=True)),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_id, logprobs_b_1)
        
        sumlogprobs_b_2 = torch.zeros(
            len(torch.unique(traj_id, sorted=True)),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_id, logprobs_b_2)
        
        # Sort rewards of done states by ascending traj id
        rewards = rewards[done.eq(1)][torch.argsort(traj_id[done.eq(1)])]
        # Trajectory balance loss
        #todo：确认是否是sumlogprobs_f_1 + sumlogprobs_f_2
        #todo：理论上单个transition是prob1*prob2，logprob1+logprob2应该没问题？
        loss = (
            (self.logZ.sum() + sumlogprobs_f_1 + sumlogprobs_f_2 - sumlogprobs_b_1 - sumlogprobs_b_2 - torch.log(rewards))
            .pow(2)
            .mean()
        )
        done_states = states[done.eq(1)][torch.argsort(traj_id[done.eq(1)])]
        return (loss, loss, loss), rewards

    #将batch中的states和trajs返回为[state1,state2..]和[(state1,state2..),(state1,state2...)]形式
    def unpack_terminal_states(self, batch):
        """
        Unpacks the terminating states and trajectories of a batch and converts them
        to Python lists/tuples.
        """
        # TODO: make sure that unpacked states and trajs are sorted by traj_id (like rewards will be)
        trajs = [[] for _ in range(self.batch_size)]
        states = [None] * self.batch_size
        for el in batch:
            traj_id = el[7][:1].item()
            state_id = el[8][:1].item()
            trajs[traj_id].append(tuple(el[1][0].tolist()))
            trajs[traj_id].append(tuple(el[2][0].tolist()))#每个元素（list）包含动作1和动作2两个元组
            
            if bool(el[6].item()):
                states[traj_id] = tuple(el[0][0].tolist())
        trajs = [tuple(el) for el in trajs]
        return states, trajs

    def train(self):
        # Metrics
        all_losses = []
        all_visited = []
        loss_term_ema = None#ema:指数平均数指标，用于计算loss_term的平均值的一种方法
        loss_flow_ema = None
        
        # Generate list of environments
        envs = [self.env.copy().reset() for _ in range(self.batch_size)]
        
        # Train loop
        #pbar进度条，起点1，终点训练步数+1，disable：是否显示进度条（True表示不显示）
        pbar = tqdm(range(1, self.n_train_steps + 1), disable=not self.logger.progress)
        for it in pbar:
            # Test
            if self.logger.do_test(it):#是否需要在当前训练步骤进行测试
                (
                    self.l1,#l1损失
                    self.kl,#kl散度
                    self.jsd,#jsd散度
                    self.corr,#相关系数
                    x_sampled,
                    kde_pred,#Kernel Density Estimate(核密度估计)的预测值
                    kde_true,
                ) = self.test(it)
                self.logger.log_test_metrics(
                    self.l1, self.kl, self.jsd, self.corr, it, self.use_context
                )
            #
            if self.logger.do_plot(it) and x_sampled is not None and len(x_sampled) > 0:
                figs = self.plot(x_sampled, kde_pred, kde_true)
                self.logger.log_plots(figs, it, self.use_context)
            
            t0_iter = time.time()#记录开始时间
            data = []#每次迭代都会清空data
            for j in range(self.sttr):#sttr：每步训练需要采样多少次，j相当于是采样轮数（每一次采样一个batch）
                batch, times = self.sample_batch(envs)#基于train中的envs（一开始为初始化的envs，每个state为source state）
                data += batch#data为总采样数据[batch1,batch2...]
            for j in range(self.ttsr):
                if self.loss == "flowmatch":
                    losses, rewards = self.flowmatch_loss(
                        it * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)#it * self.ttsr+j表示指定轮数（）
                    
                elif self.loss == "trajectorybalance":
                    losses, rewards = self.trajectorybalance_loss(
                        it * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)
                else:
                    print("Unknown loss!")
                
                #如果losses中的所有loss中存在非有限值（则跳过该迭代训练过程，无需反向传播，更新梯度，只需将最新的损失函数封装在all_losses中）    
                if not all([torch.isfinite(loss) for loss in losses]):
                    if self.logger.debug:
                        print("Loss is not finite - skipping iteration")
                    if len(all_losses) > 0:
                        all_losses.append([loss for loss in all_losses[-1]])#复制了all_losses的最后一个元素
                else:
                    losses[0].backward()#计算梯度
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )#梯度裁剪函数，self.parameters()表示待裁剪的参数列表，第二个参数是裁剪的阈值（每个参数梯度的范数阈值）
                        #由于梯度更新可能会出现梯度爆炸或梯度消失，梯度裁剪可以提高收敛，缓解上述问题
                    if self.opt is not None:#根据当前梯度和学习率等超参数，更新模型的权重
                        # required for when fp is uniform
                        self.opt.step()
                        self.lr_scheduler.step()
                        self.opt.zero_grad()#清空梯度
                    all_losses.append([i.item() for i in losses])#记录当前迭代所得损失值
                    
            # Buffer
            t0_buffer = time.time()#记录开始时间
            
            #Cost function
            states_term, trajs_term = self.unpack_terminal_states(batch)#返回states和traj的列表形式
            proxy_vals = self.env.reward2proxy(rewards).tolist()#将rewards返回为proxy的预测y值
            rewards = rewards.tolist()
            if self.logger.do_test(it) and hasattr(self.env, "get_cost"):
                costs = self.env.get_cost(states_term)#这个是用于MF-GFN的
            else:
                costs = 0.0
            # self.buffer.add(states_term, trajs_term, rewards, proxy_vals, it)
            # self.buffer.add(
            # states_term, trajs_term, rewards, proxy_vals, it, buffer="replay"
            # )
            t1_buffer = time.time()
            times.update({"buffer": t1_buffer - t0_buffer})#记录时间
            
            # Log
            if self.logger.lightweight:
                all_losses = all_losses[-100:]#只保留最近100个损失值
                all_visited = states_term#只保留当前迭代伦茨的所有states
            else:
                all_visited.extend(states_term)#将当前迭代的states_term全部添加到all_visited中
                
            # Progress bar
            self.logger.progressbar_update(
                pbar, all_losses, rewards, self.jsd, it, self.use_context
            )
            # Train logs保留每个迭代轮次it的模型权重，保存在logger中
            t0_log = time.time()
            if hasattr(self.env, "unpad_function"):
                unpad_function = self.env.unpad_function
            else:
                unpad_function = None
            self.logger.log_train(
                losses=losses,
                rewards=rewards,
                proxy_vals=proxy_vals,
                states_term=states_term,
                costs=costs,
                batch_size=len(data),
                logz=self.logZ,
                learning_rates=self.lr_scheduler.get_last_lr(),
                step=it,
                use_context=self.use_context,
                unpad_function=unpad_function,
            )
            t1_log = time.time()
            times.update({"log": t1_log - t0_log})
            # Save intermediate models
            t0_model = time.time()
            self.logger.save_models(self.forward_policy, self.backward_policy, step=it)
            t1_model = time.time()
            times.update({"save_interim_model": t1_model - t0_model})

            # Moving average of the loss for early stopping（提前训练结束，避免过拟合）
            if loss_term_ema and loss_flow_ema:#如果需要使用指数平均计算loss时
                loss_term_ema = (
                    self.ema_alpha * losses[1] + (1.0 - self.ema_alpha) * loss_term_ema
                )#指数因子*最终节点的loss+（1-指数因子）*上一次的指数平均loss
                loss_flow_ema = (
                    self.ema_alpha * losses[2] + (1.0 - self.ema_alpha) * loss_flow_ema
                )
                if (
                    loss_term_ema < self.early_stopping
                    and loss_flow_ema < self.early_stopping
                ):
                    break
            else:#不需要使用ema时
                loss_term_ema = losses[1]
                loss_flow_ema = losses[2]
            # Log times
            t1_iter = time.time()#记录结束时间（一次迭代）
            times.update({"iter": t1_iter - t0_iter})
            self.logger.log_time(times, use_context=self.use_context)

        # Save final models（forward_policy和backward_policy）
        self.logger.save_models(self.forward_policy, self.backward_policy, final=True)
        # Close logger
        if self.use_context == False:
            self.logger.end()

    def test(self, it=None):
        """
        Computes metrics by sampling trajectories from the forward policy.
        """
        if self.buffer.test_pkl is None:
            return self.l1, self.kl, self.jsd, None, None, None, None
        with open(self.buffer.test_pkl, "rb") as f:#读取测试集（轨迹）
            dict_tt = pickle.load(f)
            x_tt = dict_tt["x"]
        x_sampled, _ = self.sample_batch(self.env, self.logger.test.n, train=False)#batch,time
        if self.buffer.test_type is not None and self.buffer.test_type == "all":
            if "density_true" in dict_tt:
                density_true = dict_tt["density_true"]#读取真实的概率密度
            else:
                rewards = self.env.reward_batch(x_tt)#测试集的所有样本的reward集合
                #TODO:这里reward_batch函数只返回final state
                z_true = rewards.sum()#所有样本的reward之和
                density_true = rewards / z_true#每个样本的归一化奖励值（相当于概率密度）
                with open(self.buffer.test_pkl, "wb") as f:#将density_true写入test_pkl中
                    dict_tt["density_true"] = density_true
                    pickle.dump(dict_tt, f)
            hist = defaultdict(int)
            for x in x_sampled:
                hist[tuple(x)] += 1#统计采样样本中每个state的出现次数（一个batch中，相同的state可能会出现多次，因为其出现在不同的轨迹中）
            z_pred = sum([hist[tuple(x)] for x in x_tt]) + 1e-9#所有state的值之和+1e-9（极小值，防止除0）
            #pred：model预测出现概率（即output值），true:基于reward值的概率，目标：令每个state的出现概率与reward值成正比
            density_pred = np.array([hist[tuple(x)] / z_pred for x in x_tt])
            log_density_true = np.log(density_true + 1e-8)#1e-8防止log(0)
            log_density_pred = np.log(density_pred + 1e-8)
            kde_pred = None
            kde_true = None
        elif self.continuous:#这块不用看，是针对连续型环境
            x_sampled = torch2np(self.env.statebatch2proxy(x_sampled))
            x_tt = torch2np(self.env.statebatch2proxy(x_tt))
            kde_pred = self.env.fit_kde(
                x_sampled,
                kernel=self.logger.test.kde.kernel,
                bandwidth=self.logger.test.kde.bandwidth,
            )
            if "log_density_true" in dict_tt and "kde_true" in dict_tt:
                log_density_true = dict_tt["log_density_true"]
                kde_true = dict_tt["kde_true"]
            else:
                # Sample from reward via rejection sampling
                x_from_reward = self.env.sample_from_reward(
                    n_samples=self.logger.test.n
                )
                x_from_reward = torch2np(self.env.statetorch2proxy(x_from_reward))
                # Fit KDE with samples from reward
                kde_true = self.env.fit_kde(
                    x_from_reward,
                    kernel=self.logger.test.kde.kernel,
                    bandwidth=self.logger.test.kde.bandwidth,
                )
                # Estimate true log density using test samples
                # TODO: this may be specific-ish for the torus or not
                scores_true = kde_true.score_samples(x_tt)
                log_density_true = scores_true - logsumexp(scores_true, axis=0)
                # Add log_density_true and kde_true to pickled test dict
                with open(self.buffer.test_pkl, "wb") as f:
                    dict_tt["log_density_true"] = log_density_true
                    dict_tt["kde_true"] = kde_true
                    pickle.dump(dict_tt, f)
            # Estimate pred log density using test samples
            # TODO: this may be specific-ish for the torus or not
            scores_pred = kde_pred.score_samples(x_tt)
            log_density_pred = scores_pred - logsumexp(scores_pred, axis=0)
            density_true = np.exp(log_density_true)
            density_pred = np.exp(log_density_pred)
        else:
            raise NotImplementedError
        
        ##Metrics
        # L1 error（即MAE）
        l1 = np.abs(density_pred - density_true).mean()#计算两个概率密度的平均绝对误差（mean代表每个state的平均误差）
        # KL divergence（KL散度），度量两个概率分布相似度（用模型预测的分布来近似真实分布所造成的信息损失）
        kl = (density_true * (log_density_true - log_density_pred)).mean()
        # Jensen-Shannon divergence（JS散度）
        log_mean_dens = np.logaddexp(log_density_true, log_density_pred) + np.log(0.5)
        jsd = 0.5 * np.sum(density_true * (log_density_true - log_mean_dens))
        jsd += 0.5 * np.sum(density_pred * (log_density_pred - log_mean_dens))
        # Correlation
        if hasattr(self.env, "corr_type"):
            corr_type = self.env.corr_type
        else:
            corr_type = None
        corr = self.get_corr(density_pred, density_true, x_tt, dict_tt, corr_type)
        if hasattr(self.env, "write_samples_to_file"):
            self.env.write_samples_to_file(
                x_sampled,
                self.logger.data_path.parent
                / Path("interim_sampled_points_{}.csv".format(it)),
            )

        return (
            l1,
            kl,
            jsd,
            corr,
            x_sampled,
            kde_pred,
            kde_true,
        )

    #这块不需要
    def plot(self, x_sampled, kde_pred, kde_true, **plot_kwargs):
        # x_sampled_torch = torch.tensor(x_sampled, dtype=torch.float32)
        # fidelity = x_sampled_torch[:, -1]
        # unique, frequency = torch.unique(fidelity, dim=0, return_counts=True)
        # print("Unique samples: ", unique.shape[0])
        # print("Frequency: ", frequency)
        if hasattr(self.env, "plot_reward_samples"):
            fig_reward_samples = self.env.plot_reward_samples(x_sampled, **plot_kwargs)
        else:
            fig_reward_samples = None
        if hasattr(self.env, "plot_samples_frequency"):
            fig_samples_frequency = self.env.plot_samples_frequency(x_sampled)
        else:
            fig_samples_frequency = None
        if hasattr(self.env, "plot_reward_distribution"):
            fig_reward_distribution = self.env.plot_reward_distribution(
                states=x_sampled
            )
        else:
            fig_reward_distribution = None

        if hasattr(self.env, "plot_kde"):
            fig_kde_pred = self.env.plot_kde(kde_pred, **plot_kwargs)
            fig_kde_true = self.env.plot_kde(kde_true, **plot_kwargs)
        else:
            fig_kde_pred = None
            fig_kde_true = None

        return [
            fig_reward_samples,
            fig_kde_pred,
            fig_kde_true,
            fig_samples_frequency,
            fig_reward_distribution,
        ]

    #用于test，比较density_pred和density_true的皮尔逊相关性/
    def get_corr(self, density_pred, density_true, x_tt, dict_tt, corr_type=None):
        if corr_type == None:
            return 0.0
        if corr_type == "density_ratio":
            return np.corrcoef(density_pred, density_true)[0, 1]
        if corr_type == "from_trajectory":
            corr_matrix, _ = self.get_log_corr(x_tt, dict_tt["energy"])
            return corr_matrix[0][1]

    #每个轨迹的概率值和y值（energy）的相关性
    def get_log_corr(self, x_tt, energy):
        """
        Kept as a function variable of GflowNetAgent because logq calculation requires tfloat and tbool member functions of GflowNet
        """
        data_logq = []
        if hasattr(self.env, "_test_traj_list") and len(self.env._test_traj_list) > 0:
            for traj_list, traj_list_actions in zip(
                self.env._test_traj_list, self.env._test_traj_actions_list
            ):
                data_logq.append(
                    self.logq(
                        traj_list, traj_list_actions, self.forward_policy, self.env
                    )
                )#data_logq：每个traj的logq值
        elif hasattr(self.env, "get_trajectories"):
            test_traj_list = []
            test_traj_actions_list = []
            for state in x_tt:
                traj_list, actions = self.env.get_trajectories(
                    [],
                    [],
                    [state],
                    [(self.env.eos,)],
                )
                data_logq.append(
                    self.logq(traj_list, actions, self.forward_policy, self.env)
                )
                test_traj_list.append(traj_list)
                test_traj_actions_list.append(actions)

            setattr(self.env, "_test_traj_list", test_traj_list)
            setattr(self.env, "_test_traj_actions_list", test_traj_actions_list)
        corr = np.corrcoef(data_logq, energy)
        return corr, data_logq

    # TODO: reorganize and remove
    def log_iter(
        self,
        pbar,
        rewards,
        proxy_vals,
        states_term,
        data,
        it,
        times,
        losses,
        all_losses,
        all_visited,
    ):
        # train metrics
        self.logger.log_sampler_train(
            rewards, proxy_vals, states_term, data, it, self.use_context
        )

        # logZ
        self.logger.log_metric("logZ", self.logZ.sum(), it, use_context=False)

        # test metrics
        if not self.logger.lightweight and self.buffer.test is not None:
            corr, data_logq, times = self.get_log_corr(times)
            self.logger.log_sampler_test(corr, data_logq, it, self.use_context)

        # oracle metrics
        oracle_batch, oracle_times = self.sample_batch(
            self.env, self.oracle_n, train=False
        )

        if not self.logger.lightweight:
            self.logger.log_metric(
                "unique_states",
                np.unique(all_visited).shape[0],
                step=it,
                use_context=self.use_context,
            )

    #输入：单条轨迹；输出：指定轨迹的log_q值
    def logq(self, traj_list, actions_list, model, env, loginf=1000):
        # TODO: this method is probably suboptimal, since it may repeat forward calls for
        # the same nodes.
        log_q = torch.tensor(1.0)
        loginf = self._tfloat([loginf])
        for traj, actions in zip(traj_list, actions_list):
            traj = traj[::-1]
            actions = actions[::-1]
            masks = self._tbool(
                [env.get_mask_invalid_actions_forward(state, 0) for state in traj]
            )
            with torch.no_grad():
                logits_traj = model(self._tfloat(env.statebatch2policy(traj)))
            logits_traj[masks] = -loginf
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            logprobs_traj = logsoftmax(logits_traj)#logits→logsoftmax(logits),logsoftmax函数令每个轨迹的每个state的所有动作（包括无效动作）的对数概率logprob之和为1
            log_q_traj = torch.tensor(0.0)
            for s, a, logprobs in zip(*[traj, actions, logprobs_traj]):
                action_idx = env.action_space.index(a)#每个state的动作对应的索引值
                log_q_traj = log_q_traj + logprobs[action_idx]#log_q_traj：轨迹的概率=每个state下的实际动作的logprob之和
            # Accumulate log prob of trajectory
            if torch.le(log_q, 0.0):#if log_q<=0.0
                log_q = torch.logaddexp(log_q, log_q_traj)
            else:
                log_q = log_q_traj
        return log_q.item()#返回给定轨迹的log_q值（标量值）

#return:model(states)
class Policy:
    def __init__(
        self, 
        config, 
        env, 
        device, 
        float_precision, 
        base=None,
        count=None,#砌块库词频
        ):
        # If config is null, default to uniform
        if config is None:
            config = OmegaConf.create()
            config.type = "uniform"
        # Device and float precision
        self.device = device
        self.float = float_precision
        # Input and output dimensions
        self.state_dim_1 = env.policy_input_dim_1#state2policy_1().shape[-1]
        self.state_dim_2 = env.policy_input_dim_2#state2policy_2().shape[-1]
        
        
        self.fixed_output_1 = torch.tensor(env.fixed_policy_output_1).to(
            dtype=self.float, device=self.device
        )
        self.fixed_output_2 = torch.tensor(env.fixed_policy_output_2).to(
            dtype=self.float, device=self.device
        )
        self.output_dim_1 = len(self.fixed_output_1)
        self.output_dim_2 = len(self.fixed_output_2)
        
        self.random_output_1 = torch.tensor(env.random_policy_output_1).to(
            dtype=self.float, device=self.device
        )#所有action的logit为1
        
        self.random_output_2 = torch.tensor(env.random_policy_output_2).to(
            dtype=self.float, device=self.device
        )#所有action的logit为1
        
        #todo:policy_2参数
        self.vocab = smiles_list+["EOS"]+["UNI"]
        if count:
            self.tree = HuffmanTree(count)
        
        
        if "shared_weights" in config:#for backward_policy
            self.shared_weights = config.shared_weights
        else:
            self.shared_weights = False
            
        self.base = base#仅在backward_policy中，base为self.forward_policy
        
        if "n_hid" in config:
            self.n_hid = config.n_hid
        else:
            self.n_hid = None
            
        if "n_layers" in config:
            self.n_layers = config.n_layers
        else:
            self.n_layers = None
            
        if "tail" in config:
            self.tail = config.tail
        else:
            self.tail = []#最后一层的激活函数
            
        if "type" in config:#选择policy model的类型
            self.type = config.type
        elif self.shared_weights:#如果是backward_policy
            self.type = self.base.type
        else:
            raise "Policy type must be defined if shared_weights is False"
        
        #todo:config中添加
        if "embedding_dim_1" in config:
            self.embedding_dim_1 = config.embedding_dim_1
        else:
            self.embedding_dim_1 = 16
        if "padding_idx_1" in config:
            self.padding_idx_1 = config.padding_idx_1
        else:
            self.padding_idx_1 = 0
            
        # Instantiate policy
        if self.type == "fixed_1":
            self.model = self.fixed_distribution_1
            self.is_model = False
        elif self.type == "fixed_2":
            self.model = self.fixed_distribution_2
            self.is_model = False
        elif self.type == "random_1":
            self.model = self.random_distribution_1
            self.is_model = False
        elif self.type == "random_2":
            self.model = self.random_distribution_2
            self.is_model = False
        elif self.type == "uniform_1":
            self.model = self.uniform_distribution_1
            self.is_model = False
        elif self.type == "uniform_2":
            self.model = self.uniform_distribution_2
            self.is_model = False
        elif self.type == "policy_1":
            self.model = self.make_mlp(nn.LeakyReLU())
            self.is_model = True
        elif self.type == "policy_2":
            #todo:需要像self.mask_mlp一样定义self.shared_weights
            self.model = h_softmax(vocabulary_size = len(self.vocab),
                                    embedding_dim = self.embedding_dim_1,
                                    tree_size = len(self.tree)
                                    )
            self.is_model = True
        else:
            raise "Policy model type not defined"
        if self.is_model:
            self.model.to(self.device)
            
    def __call__(self, states):
        return self.model(states)

    def make_mlp(self, activation):
        """
        Defines an MLP with no top layer activation
        If share_weight == True,
            baseModel (the model with which weights are to be shared) must be provided
        Args
        ----
        layers_dim : list
            Dimensionality of each layer
        activation : Activation
            Activation function
        """
        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],#将之前训练好的self.base.model的除最后一层外所有层添加到了mlp的模型序列中
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),#新增最后一层，其输入维度为base.model的最后一层的输入维度，输出维度为base.model的最后一层的输出维度
            )
            return mlp.to(dtype=self.float)
        
        #如果不共享权重，那么就需要重新定义一个MLP
        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim_1] + [self.n_hid] * self.n_layers + [(self.output_dim_1)]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([activation] if n < len(layers_dim) - 2 else [])#倒数第二层才需要添加激活函数
                            for n, (idim, odim) in enumerate(
                                zip(layers_dim, layers_dim[1:])
                            )
                        ],
                        [],
                    )
                    + self.tail
                )
            )
            return mlp.to(dtype=self.float)
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )
            
    def fixed_distribution_1(self, states):
        """
        Returns the fixed distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.fixed_output_1, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )
    def fixed_distribution_2(self, states):
        """
        Returns the fixed distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.fixed_output_2, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )
    def random_distribution_1(self, states):
        """
        Returns the random distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.random_output_1, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )
    def random_distribution_2(self, states):
        """
        Returns the random distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.random_output_2, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )
    def uniform_distribution_1(self, states):
        """
        Return action logits (log probabilities) from a uniform distribution
        Args: states: tensor
        """
        return torch.ones(
            (len(states), self.output_dim_1), dtype=self.float, device=self.device
        )
    def uniform_distribution_2(self, states):
        """
        Return action logits (log probabilities) from a uniform distribution
        Args: states: tensor
        """
        return torch.ones(
            (len(states), self.output_dim_2), dtype=self.float, device=self.device
        )

class h_softmax(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int,
                tree_size: int):
        super(h_softmax, self).__init__()
        self._embedding = torch.nn.Embedding(num_embeddings=vocabulary_size,
                                             embedding_dim=embedding_dim
                                             )

        self._tree_param = torch.nn.Linear(in_features=embedding_dim,
                                           out_features=tree_size)

    
    def forward(self,
                word: torch.Tensor):
        feature = self._embedding(word)

        trans = torch.sigmoid(self._tree_param(feature))

        return trans

#优化器和学习率调整设置
def make_opt(params, logZ, config):
    """
    Set up the optimizer
    """
    params = params
    if not len(params):
        return None
    if config.method == "adam":
        opt = torch.optim.Adam(
            params,
            config.lr,
            betas=(config.adam_beta1, config.adam_beta2),#betas：用于计算梯度和梯度平方的运行平均值的系数
        )
        if logZ is not None:
            opt.add_param_group(
                {
                    "params": logZ,
                    "lr": config.lr * config.lr_z_mult,
                }
            )
    elif config.method == "msgd":
        opt = torch.optim.SGD(params, config.lr, momentum=config.momentum)
    # Learning rate scheduling
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt,#要进行学习率调整的优化器
        step_size=config.lr_decay_period,#每经过 step_size 个 epoch，学习率就会下降一次（调整为当前lr的gamma倍）
        gamma=config.lr_decay_gamma,#学习率调整的倍数
    )
    return opt, lr_scheduler