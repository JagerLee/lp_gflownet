from typing import List, Tuple
import itertools
import numpy as np
from gflownet.envs.base import GFlowNetEnv
import itertools
from polyleven import levenshtein
import numpy.typing as npt
from torchtyping import TensorType
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Sequence(GFlowNetEnv):
    """
    Anti-microbial peptide sequence environment

    Attributes
    ----------
    max_seq_length : int
        Maximum length of the sequences

    min_seq_length : int
        Minimum length of the sequences

    #字母在字母表中对应的索引数字
    nalphabet : int
        Number of letters in the alphabet
    #state：列表，表示序列，列表长度为max_seq_length，每个元素是字母表中字母的索引，从0到（nalphabet-1）
    state : list
        Representation of a sequence (state), as a list of length max_seq_length where
        each element is the index of a letter in the alphabet, from 0 to (nalphabet -
        1).

    #done：判断序列是否结束（最大长度或者执行stop action）
    done : bool
        True if the sequence has reached a terminal state (maximum length, or stop
        action executed.

    func : str
        Name of the reward function

    #应用于序列的action的数量**
    n_actions : int
        Number of actions applied to the sequence

    proxy : lambda
        Proxy model
    """

    def __init__(
        self,
        corr_type,
        max_seq_length=50,#单词最大长度
        min_seq_length=1,
        # Not required in env. But used in config_env in MLP. TODO: Find a way out
        n_alphabet=20,#字母表中字母的个数
        min_word_len=1,
        max_word_len=1,#单词最大长度
        special_tokens=None,
        proxy=None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.corr_type = corr_type
        self.lookup = {a: i for (i, a) in enumerate(self.vocab)}#字母表中字母对应的索引（i为索引，a为字母）
        self.inverse_lookup = {i: a for (i, a) in enumerate(self.vocab)}#用于依据索引获取字母
        self.n_alphabet = len(self.vocab) - len(special_tokens)#如果是碱基序列则字母表为'A','G','C','T'
        if "PAD" in self.lookup:
            self.padding_idx = self.lookup["PAD"]#padding_idx为字母表中的PAD对应的索引
        else:
            self.padding_idx = self.lookup["[nop]"]#如果字母表中没有PAD则padding_idx为字母表中的[nop]对应的索引
        # TODO: eos re-initalised in get_actions_space so why was this initialisation required in the first place (maybe mfenv)

        # self.eos = self.lookup["EOS"]#eos为字母表中的EOS对应的索引，这里不用，在获取动作列表里直接作为最终的动作元素
        self.action_space = self.get_actions_space()#获取动作空间**
        self.source = (
            torch.ones(self.max_seq_length, dtype=torch.int64) * self.padding_idx
        )#source为长度为max_seq_length的列表，每个元素为padding_idx（即初始化，每个元素都是对应pad特殊标签）
        self.reset()#重置环境
        self.fixed_policy_output = self.get_fixed_policy_output()#获取固定策略输出
        self.random_policy_output = self.get_fixed_policy_output()#获取随机策略输出
        self.policy_output_dim = len(self.fixed_policy_output)#获取策略输出维度（即参数化维度，如PF，PB）
        self.policy_input_dim = self.state2policy().shape[-1]#获取策略输入维度（即输入的state的维度）
        self.max_traj_len = self.get_max_traj_len()#获取最大轨迹长度
        # reset this to a lower value
        self.min_reward = 1e-20
        if proxy is not None:
            self.proxy = proxy#用于计算y值
        """
        规定，动作空间指代action_space
        动作列表指代动作空间中每个元素的长度的汇总列表length_of_action
        """
        length_of_action = [len(a) for a in self.action_space]#length_of_action为列表，列表长度为动作空间的长度，每个元素为对应单词长度的动作元组数量
        self.length_of_action = torch.tensor(length_of_action)#将该列表转换为tensor

    #构建actions_space（一个列表，记录了所有可能的动作），返回动作列表（每个长度的单词对应的所有动作元组作为一个列表元素）
    def get_actions_space(self):
        """
        Constructs list with all possible actions
        If min_word_len = n_alphabet = 2, actions: [(0, 0,), (1, 1)] and so on
        """
        assert self.max_word_len >= self.min_word_len
        valid_wordlens = np.arange(self.min_word_len, self.max_word_len + 1)#创建整数列表，数字范围为[min_word_len,max_word_len]
        alphabet = [a for a in range(self.n_alphabet)]#alphabet为列表，记录单词字母对应的数字（即这里记录的是数字），如[0, 1, 2, 3, 4]
        actions = []#创建actions列表,其记录形式为[元素1，元素2]，元素1为列表，记录了单词长度为x下的所有动作组合元组可能
        """
        举例：
        假设alphabet=[0, 1, 2],r为2，则actions_r=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1),(1, 2), (2, 0), (2, 1), (2, 2)]
        """
        for r in valid_wordlens:#r为单词长度，单词有多长，就有多少个动作（r=n_action）
            actions_r = [el for el in itertools.product(alphabet, repeat=r)]#对于长度为r的单词，有n_alphabet^r个动作
            ###这里我需要改进，这里的n_alphabet^r指的是r步的总动作记录，实际上每一步的实际可用动作是小于n_alphabet的
            actions += actions_r
        # Add "eos" action
        # eos != n_alphabet in the init because it would break if max_word_len >1
        actions = actions + [(len(actions),)]#在actions列表中添加一个元素
        """
        如[[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)], 
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2)], 
        (2,)]这里最后的这个(2,)就是eos，其索引为len(actions)-1
        """
        self.eos = len(actions) - 1#eos为actions列表中最后一个元素的索引数字
        return actions

    #这个函数需要再修改
    #输入的是单个state（数组形式，如[1,2,0,1])
    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space (where action space includes eos): True if action is invalid
        given the current state, False otherwise.
        #返回一个长度为动作空间的向量（其中动作空间包括eos）：如果当前状态下动作无效则为True，否则为False。
        """
        if state is None:
            state = self.state.clone().detach()#返回state的副本，且不会被记录在计算图中
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space))]#如果当前序列已经结束，返回一个全为True的列表
        # mask = [False for _ in range(len(self.action_space))]
        
        #seq_length存储了输入state的序列长度（若无pad，则为state完整长度；若有pad，则为state中第一个pad索引的长度）
        seq_length = (
            torch.where(state == self.padding_idx)[0][0]#首先判断 state 中是否存在 padding_idx（即填充标记）
            #如果 state 的最后一个元素为 padding_idx，则该序列的实际长度就是第一个出现 padding_idx 的位置（即在张量中第一次出现padding_idx的索引）
            
            if state[-1] == self.padding_idx#如果state的最后一个元素为padding_idx
            else len(state)#否则，这个序列的实际长度就是 state 张量的长度
        )
        # # set mask to True for all actions that would exceed max_seq_length
        
        #seq_length和self.max_seq_length都用于记录输入的单个state的动作空间（不包含ens）
        seq_length_tensor = torch.ones(
            (len(self.action_space[:-1]),), dtype=torch.int64
        ) * (seq_length)#创建了一个维度为len(self.action_space[:-1])，元素均为seq_length的张量（相当于维度数是单词长度类型数，每个维度的元素为state的有效长度）
        
        updated_seq_length = seq_length_tensor + self.length_of_action[:-1]#updated_seq_length是上述两个张量的逐维度元素相加（除了length_of_action的最后一个eos）
        #首先比较updated_seq_length和max_seq_length。
        #如果updated_seq_length大于max_seq_length，则返回True，否则是False。这样，所有大于max_seq_length的序列位置都将被标记为True。
        mask = updated_seq_length > self.max_seq_length#逐位置（updated_seq_length的每个单词长度类型维度）与最大序列长度比较，mask为np数组（大于最大序列长度的位置为True，否则为False）
        mask = mask.tolist()#将数组mask转换为列表形式
        if seq_length < self.min_seq_length:#如果序列有效长度小于设定的最小序列长度，则在mask列表最后添加一个True（表示需要mask)
            mask.append(True)
        else:
            mask.append(False)
        # for idx, a in enumerate(self.action_space[:-1]):
        #     if seq_length + len(list(a)) > self.max_seq_length:
        #         mask[idx] = True
        return mask
        #mask列表将包含updated_seq_length中所有超过max_seq_length的位置和一个额外的True/False（用于表示state有效长度是否小于最小序列长度）。
        """
        举例：假设当前state的seq_length(有效长度)为1，actions_space为[[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)], 
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2)], 
        (2,)]
        则seq_length_tensor为tensor([1,1])
        length_of_action为[9,27,1](9，27分别表示单词长度为1和2)
        updated_seq_length为tensor([10,28])
        mask = updated_seq_length > self.max_seq_length#mask为tensor([False, False])
        mask = mask.tolist()#mask变为[False, False]
        所以mask的元素数取决于动作空间的单词长度类型数
        #这里的mask逻辑需要修改？
        """
        
        
        
    def true_density(self, max_states=1e6):#这里的1e6指的是final_state的数量
        """
        Computes the reward density (reward / sum(rewards)) of the whole space, if the
        dimensionality is smaller than specified in the arguments.
        #计算整个空间的奖励密度（指代奖励/总奖励之和），如果维度小于参数中指定的维度。
        Returns
        -------
        Tuple:
          - normalized reward for each state#每个state的归一化奖励
          - states#states列表
          - (un-normalized) reward)#每个state的（非归一化）奖励
        """
        if self._true_density is not None:
            return self._true_density
        if self.n_alphabet**self.max_seq_length > max_states:
            return (None, None, None)
        
        #获取所有的state（np数组形式，每个元素都是一个列表（代表了一种state，长度为最大序列长度）
        """
        如array([[0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 0],
            [0, 2, 1],
            ...
            [3, 3, 3]])
        """
        state_all = np.int32(
            list(
                itertools.product(*[list(range(self.n_alphabet))] * self.max_seq_length)
            )
        )
        
        #对输入的state_all数组中的指定state进行处理，将结果分别存储在两个元组traj_rewards和state_end中
        traj_rewards, state_end = zip(
            *[
                (self.proxy(state), state)#zip打包为元组（奖励值，原state）
                for state in state_all#对于满足以下条件的state进行上述处理
                if len(self.get_parents(state, False)[0]) > 0 or sum(state) == 0#如果某个状态的父节点非空或者该状态的所有元素之和为 0，则选择该状态进行处理。
            ]
        )
        traj_rewards = np.array(traj_rewards)#将traj_rewards转换为np数组形式
        self._true_density = (
            traj_rewards / traj_rewards.sum(),
            list(map(tuple, state_end)),#将state转换为元组形式，并存储在一个列表中
            traj_rewards,#非归一化奖励（每个state的）
        )#_true_density相当于一个三元素的元组，元素一表示traj_rewards中每个state的归一化奖励
        return self._true_density

    # def state2oracle(self, state: List = None):
    #     return "".join(self.state2readable(state))

    #返回最大轨迹长度（但是也可以自行设置，这里我感觉有点问题，word的定义需要再明确）
    def get_max_traj_len(
        self,
    ):
        return self.max_seq_length / self.min_word_len + 1#返回最大轨迹长度，即最大序列长度/最小单词长度+1

    #一批states转换为oracle形式（如'AGGGCT'），输入对象为list形式的states
    def statebatch2oracle(
        self, states: List[TensorType["max_seq_length"]]
    ) -> List[str]:
        state_oracle = []
        for state in states:
            if state[-1] == self.padding_idx:
                state = state[: torch.where(state == self.padding_idx)[0][0]]
            if self.tokenizer is not None and state[0] == self.tokenizer.bos_idx:
                state = state[1:-1]
            state_numpy = state.detach().cpu().numpy()#将state转换为np数组形式
            state_oracle.append(self.state2oracle(state_numpy))#将state转换为oracle形式
        return state_oracle

    #一批states转换为oracle形式（如'AGGGCT'），输入对象为tensor形式的states
    def statetorch2oracle(
        self, states: TensorType["batch_dim", "max_seq_length"]
    ) -> List[str]:
        return self.statebatch2oracle(states)

    #将state转换为one-hot编码列表
    #如[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
    #  |     A    |      T    |      G    |      C    |
    #我的字母表有10w+个字母，因此单个字母（即一个smiles）的one-hot编码就有10w+个元素
    #如果我先基于smiles制作smiles词表，大概就20个元素，这样单个state的长度也就最多20*3（三个smiles组合）
    def state2policy(self, state=None):
        """
        Transforms the sequence (state) given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of length nalphabet * max_seq_length,
        where each n-th successive block of nalphabet elements is a one-hot encoding of
        the letter in the n-th position.
        #将作为参数给出的序列（state）（如果为None，则为self.state）转换为one-hot编码。
        #输出是一个长度为nalphabet * max_seq_length的列表，其中每个第n个连续的nalphabet元素块是第n个位置的字母的one-hot编码。

        #nalphabet * max_seq_length即为完全一维展开，如下面的示例所示
        
        Example:
          - Sequence: AATGC
          - state: [0, 1, 3, 2]
                    A, T, G, C
          - state2obs(state): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                              |     A    |      T    |      G    |      C    |

        If max_seq_length > len(state), the last (max_seq_length - len(state)) blocks are all
        0s.
        """

        if state is None:
            state = self.state.clone().detach()
        state_onehot = (
            F.one_hot(state, num_classes=self.n_alphabet + 2)[:, :-2]
            .to(self.float)
            .to(self.device)
        )
        state_onehot = state_onehot.unsqueeze(0)
        state_policy = torch.zeros(1, self.max_seq_length, self.n_alphabet)#生成一个全零张量
        state_policy[:, : state_onehot.shape[1], :] = state_onehot#将state_onehot的值赋给state_policy
        return state_policy.reshape(1, -1)

    #将一批states转换为policy模型的输入形式，输入对象为list形式的states
    def statebatch2policy(
        self, states: List[TensorType["1", "max_seq_length"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Transforms a batch of states into the policy model format. The output is a numpy
        array of shape [n_states, n_alphabet * max_seq_len].

        See state2policy()
        """
        state_tensor = torch.vstack(states)
        state_policy = self.statetorch2policy(state_tensor)
        return state_policy

    #输入：一批states（tensor形式），输出为其onehot编码形式
    def statetorch2policy(
        self, states: TensorType["batch", "max_seq_length"]
    ) -> TensorType["batch", "policy_input_dim"]:
        if states.dtype != torch.long:
            states = states.to(torch.long)
        state_onehot = (
            F.one_hot(states, self.n_alphabet + 2)[:, :, :-2]
            .to(self.float)
            .to(self.device)
        )
        state_padding_mask = (states != self.padding_idx).to(self.float).to(self.device)#生成一个与states相同大小的，布尔型的张量，值为True的位置对应states中非pad索引的位置
        state_onehot_pad = state_onehot * state_padding_mask.unsqueeze(-1)
        #state_onehot为states的one-hot编码，
        #*state_padding_mask.unsqueeze(-1)会在states_onehot的最后一个维度添加states大小一样的state_padding_mask张量，与state_onehot相乘后，填充位置的值全部归0

        assert torch.eq(state_onehot_pad, state_onehot).all()
        state_policy = torch.zeros(
            states.shape[0],
            self.max_seq_length,
            self.n_alphabet,
            device=self.device,
            dtype=self.float,
        )
        state_policy[:, : state_onehot.shape[1], :] = state_onehot
        return state_policy.reshape(states.shape[0], -1)#states.shape[0]即batch大小

    #将onehot编码形式（state_policy)转换为state的np数组形式(如[0, 0, 1, 3, 2])
    def policytorch2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AATGC
          - state_policy: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                          |     A    |      A    |      T    |      G    |      C    |
          - policy2state(state_policy): [0, 0, 1, 3, 2]
                    A, A, T, G, C
        """
        mat_state_policy = torch.reshape(
            state_policy, (self.max_seq_length, self.n_alphabet)
        )
        state = torch.where(mat_state_policy)[1].tolist()
        return state

    # TODO: Deprecate as never used.弃用
    def policy2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AATGC
          - state_policy: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                 |     A    |      A    |      T    |      G    |      C    |
          - policy2state(state_policy): [0, 0, 1, 3, 2]
                    A, A, T, G, C
        """
        mat_state_policy = np.reshape(
            state_policy, (self.max_seq_length, self.n_alphabet)
        )
        state = np.where(mat_state_policy)[1].tolist()
        return state

    #即将数值数组形式的序列（state)转换为字符串形式，如'AGGGCT'
    def state2oracle(self, state: List = None):
        return "".join(self.state2readable(state))

    #将数值数组形式的序列（state)转换为字母序列（称之为readable）（str形式，如'AGGGCT'）
    def state2readable(self, state: List) -> str:
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        Used only in Buffer
        """
        if isinstance(state, torch.Tensor) == False:
            state = torch.tensor(state).long()
        if state[-1] == self.padding_idx:
            state = state[: torch.where(state == self.padding_idx)[0][0]]
        state = state.tolist()
        return "".join([self.inverse_lookup[el] for el in state])

    #输入state的Tensor size为(1,max_seq_length)，输出为str形式的序列（如'AGGGCT'）
    def statetorch2readable(self, state: TensorType["1", "max_seq_length"]) -> str:
        if state[-1] == self.padding_idx:
            state = state[: torch.where(state == self.padding_idx)[0][0]]
        # TODO: neater way without having lookup as input arg
        if (
            self.lookup is not None
            and "[CLS]" in self.lookup.keys()
            and state[0] == self.lookup["[CLS]"]
        ):
            state = state[1:-1]#cls是常用的bert的特殊标签，这里是为了去除可能有的cls标签
        state = state.tolist()
        readable = [self.inverse_lookup[el] for el in state]
        return "".join(readable)

    def readable2state(self, readable) -> TensorType["batch_dim", "max_seq_length"]:
        """
        Transforms a list or string of letters into a list of indices according to an alphabet.
        """
        if isinstance(readable, str):
            encoded_readable = [self.lookup[el] for el in readable]
            state = (
                torch.ones(self.max_seq_length, dtype=torch.int64) * self.padding_idx
            )
            state[: len(encoded_readable)] = torch.tensor(encoded_readable)
        else:
            encoded_readable = [[self.lookup[el] for el in seq] for seq in readable]
            state = (
                torch.ones((len(readable), self.max_seq_length), dtype=torch.int64)
                * self.padding_idx
            )
            for i, seq in enumerate(encoded_readable):
                state[i, : len(seq)] = torch.tensor(seq)
        return state

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = (
            torch.ones(self.max_seq_length, dtype=torch.int64) * self.padding_idx
        )
        self.done = False
        self.id = env_id
        self.n_actions = 0
        return self

    #return parents, actions
    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to sequence state

        Args
        ----
        state : list
            Representation of a sequence (state), as a list of length max_seq_length
            where each element is the index of a letter in the alphabet, from 0 to
            (nalphabet - 1).

        action : int
            Last action performed, only to determine if it was eos.

        Returns
        -------
        parents : list
            List of parents as state2obs(state)

        actions : list
            List of actions that lead to state for each parent in parents
        """
        # TODO: Adapt to tuple form actions
        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [state], [(self.eos,)]
        elif torch.eq(state, self.source).all():
            return [], []
        else:
            parents = []
            actions = []
            if state[-1] == self.padding_idx:
                state_last_element = int(torch.where(state == self.padding_idx)[0][0])
            else:
                state_last_element = len(state)
            max_parent_action_length = self.max_word_len + 1 - self.min_word_len
            for parent_action_length in range(1, max_parent_action_length + 1):
                parent_action = tuple(
                    state[
                        state_last_element - parent_action_length : state_last_element
                    ].numpy()
                )
                if parent_action in self.action_space:
                    parent = state.clone().detach()
                    parent[
                        state_last_element - parent_action_length : state_last_element
                    ] = self.padding_idx
                    parents.append(parent)
                    actions.append(parent_action)
        return parents, actions

    #输入给定的动作元组（针对一个完整单词序列的），返回三元素元组（执行动作后的最终state；动作元组；布尔值表示当前动作是否有效），即self.state, action, valid
    #注意，这里的action应该是一个完整序列的动作数组
    def step(self, action: Tuple[int]) -> Tuple[List[int], Tuple[int, int], bool]:
        """
        Executes step given an action index
        #基于给定的动作索引执行步骤
        
        If action_idx is smaller than eos (no stop), add action to next
        position.#如果action_idx小于eos（没有停止），则将action添加到下一个位置。

        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"
        #动作空间中动作的索引。a == eos表示“停止动作”
        
        Returns
        -------
        self.state : list
            The sequence after executing the action#执行动作后的序列

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        assert action in self.action_space
        # If only possible action is eos, then force eos
        if self.state[-1] != self.padding_idx:#如果state（数组形式）的最后一个索引不是填充索引，说明当前state是最终状态
            self.done = True
            self.n_actions += 1#可用动作数+1
            return self.state, (self.eos,), True
        # If action is not eos, then perform action#如果动作不是eos，则执行动作
        state_last_element = int(torch.where(self.state == self.padding_idx)[0][0])#表示最后一个有效动作（第一个填充动作）的索引
        if action[0] != self.eos:#如果动作不是eos（因为eos的第一个数值是唯一的，对应self.eos，所以只需要判断第一个数值是否是eos即可）
            state_next = self.state.clone().detach()#复制当前state
            if state_last_element + len(action) > self.max_seq_length:
                valid = False
            else:
                state_next[
                    state_last_element : state_last_element + len(action)
                ] = torch.LongTensor(action)#state_next即添加一个动作后的下一个state（数组形式）
                self.state = state_next#更新当前state
                valid = True
                self.n_actions += 1#可用动作数+1（指当前完整动作序列是有效的）
            return self.state, action, valid
        else:
            if state_last_element < self.min_seq_length:
                valid = False
            else:
                self.done = True
                valid = True
                self.n_actions += 1
            return self.state, (self.eos,), valid

    def get_pairwise_distance(self, samples, *kwargs):
        dists = []
        for pair in itertools.combinations(samples, 2):
            distance = self.get_distance(*pair)
            dists.append(distance)
        dists = torch.FloatTensor(dists)
        return dists

    def get_distance_from_D0(self, samples, dataset_states):
        # TODO: optimize
        dataset_samples = self.statetorch2oracle(dataset_states)
        min_dists = []
        for sample in samples:
            dists = []
            sample_repeated = itertools.repeat(sample, len(dataset_samples))
            for s_0, x_0 in zip(sample_repeated, dataset_samples):
                dists.append(self.get_distance(s_0, x_0))
            min_dists.append(np.min(np.array(dists)))
        return torch.FloatTensor(min_dists)

    def get_distance(self, seq1, seq2):
        return levenshtein(seq1, seq2) / 1
