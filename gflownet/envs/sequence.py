from typing import List, Tuple
import itertools
import numpy as np
from gflownet.envs.base import GFlowNetEnv
#from polyleven import levenshtein
import numpy.typing as npt
from torchtyping import TensorType
import torch
#import matplotlib.pyplot as plt
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import random
from gflownet.data.template import smarts_list, name_list 


class Sequence(GFlowNetEnv):
    """
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

    #应用于序列的action的数量，用于计算轨迹中的state序号
    n_actions : int
        Number of actions applied to the sequence

    proxy : lambda
        Proxy model
    """

    def __init__(
        self,
        corr_type,
        max_seq_length=7,#序列最大长度为5，砌块1，砌块2，砌块3，反应1，反应1产物，反应2，反应2产物
        min_seq_length=1,
        n_alphabet=97967,#砌块库砌块数量，后续会根据使用的enamine砌块库类型/反应模板类型/数据处理手段进行修改
        special_tokens_1=None,
        special_tokens_2=None,
        proxy=None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.corr_type = corr_type
        self.lookup_1={a: i for (i, a) in enumerate(self.vocab_1)}#基于反应模板smarts+特殊字符(注意，只包含eos)的列表vocab_1构建字典
        self.inverse_lookup_1 = {i: a for (i, a) in enumerate(self.vocab_1)}#用于依据索引获取字母
        self.lookup_2 = {a: i for (i, a) in enumerate(self.vocab_2)}#字典构建，vocab_2=砌块库列表+eos+uni+pad
        self.inverse_lookup_2 = {i: a for (i, a) in enumerate(self.vocab_2)}#用于依据索引获取字母

        self.n_alphabet_1 = len(self.vocab_1) - len(special_tokens_1)#反应smarts数量
        self.n_alphabet_2 = len(self.vocab_2) - len(special_tokens_2)#砌块库砌块数量
        
        self.padding_idx = self.lookup_2["PAD"]#padding_idx为字母表中的PAD对应的索引
        self.uni_idx = self.lookup_2["UNI"]#代表单组分，无需添加砌块
        self.eos_1 = self.lookup_1["EOS"]
        self.eos_2 = self.lookup_2["EOS"]
        
        self.action_space_1 = self.get_actions_space_1()#动作空间1:指示反应模板索引
        self.action_space_2 = self.get_actions_space_2()#动作空间2：指示砌块索引
        
        self.source = (
            torch.ones(self.max_seq_length, dtype=torch.int64) * self.padding_idx
        )#initial_state,每个元素对应的都是pad标签
        
        self.reset()#重置环境
        
        self.fixed_policy_output_1 = self.get_fixed_policy_output_1()
        self.fixed_policy_output_2 = self.get_fixed_policy_output_2()
        
        self.random_policy_output_1 = self.get_fixed_policy_output_1()
        self.random_policy_output_2 = self.get_fixed_policy_output_2()
        
        self.policy_output_dim_1 = len(self.fixed_policy_output_1)
        self.policy_output_dim_2 = len(self.fixed_policy_output_2)
        
        self.smarts_list = smarts_list
        self.name_list = name_list
        self.uni_list = [8, 49, 50, 63, 64, 79, 80, 81, 82, 83, 84, 85, 86] #单组分反应索引形式
        
        self.policy_input_dim_1 = self.state2policy_1().shape[-1]
        self.policy_input_dim_2 = self.state2policy_2().shape[-1]
        
        self.max_traj_len = self.get_max_traj_len()#获取最大轨迹长度，可舍弃
        # reset this to a lower value
        self.min_reward = 1e-20
        if proxy is not None:
            self.proxy = proxy
            
        length_of_action_1 = [len(a) for a in self.action_space_1]#记录每个动作（元组形式）的长度
        length_of_action_2= [len(a) for a in self.action_space_2]
        self.length_of_action_1 = torch.tensor(length_of_action_1)
        self.length_of_action_2 = torch.tensor(length_of_action_2)

        
    #返回动作列表1，包含反应模板smarts+eos
    def get_actions_space_1(self):
        alphabet = [a for a in range(self.n_alphabet_1)]#反应模板的索引表，0-n_alphabet-2
        actions = [el for el in itertools.product(alphabet, repeat=1)]#eg:[(0,), (1,), (2,), (3,), (4,)]
        actions = actions + [(len(actions),)]#在actions列表中添加一个元素（代表eos）eg:[(0,), (1,), (2,), (3,), (4,), (5,)]
        return actions
    
    #返回动作列表2，包含砌块+eos+uni
    def get_actions_space_2(self):
        alphabet = [a for a in range(self.n_alphabet_2)]#砌块库的索引表，0-n_alphabet-1
        actions = [el for el in itertools.product(alphabet, repeat=1)]#eg:[(0,), (1,), (2,), (3,), (4,)]
        actions = actions + [(len(actions),)]+[(len(actions)+1,)]#在actions列表中添加两个元素（分别代表eos和uni）eg:[(0,), (1,), (2,), (3,), (4,), (5,)]
        return actions
    
    #判断输入的smiles是否可以作为该反应模板smarts的反应物，返回布尔值 
    def check_mask_1(self,smiles,smarts):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            return True
        rxn = AllChem.ReactionFromSmarts(smarts)
        reactants = rxn.GetReactants()
        #如果是双组分反应
        if len(reactants) == 2:
            if mol.HasSubstructMatch(reactants[0]) or mol.HasSubstructMatch(reactants[1]):
                return False
        #如果是单组分反应
        else:
            if mol.HasSubstructMatch(reactants[0]):
                return False
        return True
    
    #maks_1:判断输入的state（砌块/中间产物）可以基于action_1中的哪些反应模板进行反应
    #O(模板数)
    def get_mask_invalid_actions_forward_1(self, state=None, done=None):
        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space_1))]#如果当前序列已经结束，返回一个全为True的列表
        
        alphabet = [a for a in range(self.n_alphabet_1)]#反应模板索引
        smarts_list = [self.inverse_lookup_1[x] for x in alphabet]#[smarts1,smart2,...]反应模板列表
        
        mask = [False for _ in range(len(self.action_space_1))]#初始化，所有动作都不mask(包括eos动作)
        
        for idx, _ in enumerate(self.action_space_1[:-1]):
            if self.check_mask_1(self.state2readable(state),smarts_list[idx]):
                mask[idx] = True
        return mask
    
    #返回双组分反应模板时的对应key
    def check_mask_2(self,smiles,smarts,index):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            return True
        rxn = AllChem.ReactionFromSmarts(smarts)
        reactants = rxn.GetReactants()    
        if mol.HasSubstructMatch(reactants[0]):#如果已有分子是反应物1，则待选择砌块必须是反应物2
            return self.name_list[index]+"_reactant_2"
        if mol.HasSubstructMatch(reactants[1]):
            return self.name_list[index]+"_reactant_1"
                
    #mask_2:判断输入的state（砌块/中间产物）与action_1(选择的反应模板)与哪些砌块可以反应,返回的是一个字符串key
    #todo:需要考虑动作1选到终止动作的情况吗？
    def get_mask_invalid_actions_forward_2(self, state=None, done=None, action_1=None):
        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space_2))]#如果当前序列已经结束，返回一个全为True的列表
        #如果输入state为初始state
        if state[0] == self.padding_idx:
            mask = 'any'  
            return mask
        #如果采样的模板为单组分反应
        if action_1[0] in self.uni_list:
            mask = 'uni'
            return mask
        else:
            alphabet = [a for a in range(self.n_alphabet_2)]#砌块库索引
            smiles_list = [self.inverse_lookup_2[x] for x in alphabet]#['smiles1','smiles'...]砌块的smiles列表
            smarts = self.inverse_lookup_1[action_1[0]]#采样的反应模板的smarts形式
            index = action_1[0]
            mask = self.check_mask_2(self.state2readable(state),smarts,index)
            return mask

    #可以舍弃    
    def true_density(self, max_states=1e6):#这里的1e6指的是final_state的数量
        if self._true_density is not None:
            return self._true_density
        if self.n_alphabet_1**self.max_seq_length > max_states:
            return (None, None, None)
        
        #获取所有的state（np数组形式，每个元素都是一个列表（代表了一种state，长度为最大序列长度）
        state_all = np.int32(
            list(
                itertools.product(*[list(range(self.n_alphabet_1))] * self.max_seq_length)
            )
        )
        
        #对输入的state_all数组中的指定state进行处理，将结果分别存储在两个元组traj_rewards和state_end中
        traj_rewards, state_end = zip(
            *[
                (self.proxy(state), state)
                for state in state_all#对于满足以下条件的state进行上述处理
                if len(self.get_parents(state, False)[0]) > 0 or sum(state) == 0#
            ]
        )
        traj_rewards = np.array(traj_rewards)
        self._true_density = (
            traj_rewards / traj_rewards.sum(),
            list(map(tuple, state_end)),#将state转换为元组形式，并存储在一个列表中
            traj_rewards,#非归一化奖励（每个state的）
        )#_true_density相当于一个三元素的元组，元素一表示traj_rewards中每个state的归一化奖励
        
        return self._true_density

    #返回最大轨迹长度（序列长度+1，1对应initial_state）
    def get_max_traj_len(
        self,
    ):
        return self.max_seq_length + 1#返回最大轨迹长度

    #将砌块/中间产物/终产物的smiles转换为policy_1的输入形式
    #如果输入的是初始状态，返回一个全0tensor作为policy_1的输入#todo：有无更好的方法，不让其进行action_1的选择
    #todo:当前先用morgan指纹对分子进行表征
    def state2policy_1(self,state=None):
        if state is None:
            state = self.state.clone().detach()
        state = state.tolist()
        #如果state是初始state，返回一个全0张量
        if state[0] == self.padding_idx:
            state_policy = torch.zeros(4096)
        else:
            state_readable = self.state2readable(state)#product_smiles
            mol = Chem.MolFromSmiles(state_readable)
            features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol,2,4096)
            features_vec = np.array(features_vec)
            state_policy = torch.tensor(features_vec.reshape((1, -1)), dtype=self.float)
        return state_policy
    
    #将反应模板作为label（smarts形式）转换为onehot形式
    def label2onehot(self,label:str):
        label_to_index = {label: i for (i,label) in enumerate(self.smarts_list)}
        label_onehot = torch.zeros(len(self.smarts_list))
        index = label_to_index[label]
        label_onehot[index] = 1.0
        return label_onehot    
    
    #1.动作1为空(选择第一个砌块时); 2.动作1为模板
    def state2policy_2(self,state=None,action_1=None):
        if state is None:
            state = self.state.clone().detach()
        state = state.tolist()
        if action_1 is None:
            label_onehot = torch.zeros(len(self.smarts_list))
            features_vec = torch.zeros(4096)
            state_policy = torch.cat((features_vec.reshape(1,-1),label_onehot.reshape(1,-1)),dim=1).type(self.float)
        else:
            state_readable = self.state2readable(state)#product_smiles
            mol = Chem.MolFromSmiles(state_readable)
            features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol,2,4096)
            features_vec = torch.tensor(np.array(features_vec),dtype=self.float)
            action_1 = tuple(action_1.tolist())
            label_onehot = self.label2onehot(self.inverse_lookup_1[action_1[0]])
            state_policy = torch.cat((features_vec.reshape(1,-1),label_onehot.reshape(1,-1)),dim=1).type(self.float)
        return state_policy
    
    #
    def statetorch2policy_1(self, states: TensorType["batch", "max_seq_length"]):
        if states.dtype != torch.long:
            states = states.to(torch.long)
        batch_size = states.shape[0]
        
        state_policy_list = []
        for i in range(batch_size):
            state_policy = self.state2policy_1(states[(i,)])
            state_policy_list.append(state_policy)
        state_policy_batch = torch.cat(state_policy_list, dim=0)
        return state_policy_batch.reshape(batch_size, -1)

    #
    def statebatch2policy_1(
        self, states: List[TensorType["1", "max_seq_length"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        state_tensor = torch.vstack(states)#将states列表转换为张量形式
        state_policy = self.statetorch2policy_1(state_tensor)
        return state_policy
    #
    def statetorch2policy_2(self, states: TensorType["batch", "max_seq_length"],actions_1: TensorType["batch", "2"]):
        if states.dtype != torch.long:
            states = states.to(torch.long)
        batch_size = states.shape[0]
        
        state_policy_list = []
        for i in range(batch_size):
            state_policy = self.state2policy_2(states[(i,)],actions_1[(i,)])
            state_policy_list.append(state_policy)
        state_policy_batch = torch.cat(state_policy_list, dim=0)
        return state_policy_batch.reshape(batch_size, -1)
    #
    def statebatch2policy_2(
        self, states: List[TensorType["1", "max_seq_length"]],actions_1: List[TensorType["1", "2"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        state_tensor = torch.vstack(states)#将states列表转换为张量形式
        actions_1_tensor = torch.vstack(actions_1)#将动作列表转换为tensor形式#todo
        state_policy = self.statetorch2policy_2(state_tensor,actions_1_tensor)
        return state_policy
    
    #state从数值列表转换为product_smiles
    def state2oracle(self, state: List = None):
        return self.state2readable(state)
    
    def statetorch2oracle(
        self, states: TensorType["batch_dim", "max_seq_length"]
    ) -> List[str]:
        return self.statebatch2oracle(states) 
    
    def statebatch2oracle(
        self, states: List[TensorType["max_seq_length"]]
    ) -> List[str]:
        state_oracle = []
        for state in states:
            if state[-1] == self.padding_idx:
                state = state[: torch.where(state == self.padding_idx)[0][0]]#转换为有效state
            if self.tokenizer is not None and state[0] == self.tokenizer.bos_idx:#如果state的第一个元素为bos_idx
                state = state[1:-1]
            state_numpy = state.detach().cpu().numpy()#将state转换为np数组形式
            state_oracle.append(self.state2oracle(state_numpy))#[product_smiles1,product_smiles2,...]
        return state_oracle

    #获取单组分反应的产物index
    def get_uni_product_idx(self,smiles,action_1):
        mol = Chem.MoleFromSmiles(smiles)
        rxn = AllChem.ReactionFromSmarts(self.inverse_lookup_1[action_1])
        ps = rxn.RunReactants((mol,))
        uniqps = {}
        for p in ps:
            Chem.SanitizeMol(p[0])
            inchi = Chem.MolToInchi(p[0])
            uniqps[inchi] = Chem.MolToSmiles(p[0])
        uniqps_sort = sorted(uniqps.values())
        return random.randrange(len(uniqps_sort))
  
    #输入state[idx]，action_1,action_2，返回双组分product_idx
    def get_bi_product_idx(self,smiles,action_1,action_2):
        mol_1 = Chem.MolFromSmiles(smiles)
        mol_2 = Chem.MolFromSmiles(self.inverse_lookup_2[action_2])
        rxn = AllChem.ReactionFromSmarts(self.inverse_lookup_1[action_1])
        ps = rxn.RunReactants((mol_1,mol_2))+rxn.RunReactants((mol_2,mol_1))
        uniqps = {}
        for p in ps:
            Chem.SanitizeMol(p[0])
            inchi = Chem.MolToInchi(p[0])
            uniqps[inchi] = Chem.MolToSmiles(p[0])
        uniqps_sort=sorted(uniqps.values())#产物smiles列表
        #从列表中随机获取一个index作为product_idx
        return random.randrange(len(uniqps_sort))
        
    #双组分反应，基于两个smiles和反应smarts以及产物列表对应的索引，返回产物的smiles——用于state2readable
    def get_bi_product(self,smiles_1,smiles_2,smarts,index):
        mol_1 = Chem.MolFromSmiles(smiles_1)
        mol_2 = Chem.MolFromSmiles(smiles_2)
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((mol_1,mol_2))+rxn.RunReactants((mol_2,mol_1))
        uniqps = {}
        for p in ps:
            Chem.SanitizeMol(p[0])
            inchi = Chem.MolToInchi(p[0])
            uniqps[inchi] = Chem.MolToSmiles(p[0])
        uniqps_sort = sorted(uniqps.values())
        smiles = uniqps_sort[index]
        return smiles
    
    #单组分反应，基于一个smiles和反应smarts以及产物列表对应的索引返回产物smiles
    def get_uni_product(self,smiles1,smarts,index):
        mol = Chem.MolFromSmiles(smiles1)
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((mol,))
        uniqps = {}
        for p in ps:
            Chem.SanitizeMol(p[0])
            inchi = Chem.MolToInchi(p[0])
            uniqps[inchi] = Chem.MolToSmiles(p[0])
        uniqps_sort = sorted(uniqps.values())
        smiles = uniqps_sort[index]
        return smiles
    
    #todo:state_readable:['smiles','smiles','smiles',rxn1,ps_1,rxn2,ps_2],len=7
    #用于state2policy和state2proxy，返回产物smiles
    def state2readable(self, state: List) -> str:
        if isinstance(state, torch.Tensor) == False:#如果state不是张量形式，则转换为张量形式
            state = torch.tensor(state).long()        
            
        if state[0] == self.padding_idx:#如果为初始state，返回None
            return None
        elif state[1] == self.padding_idx:#如果只有第一个砌块，返回该砌块smiles
            smiles = self.inverse_lookup_2[state[0]]
            return smiles
        #两个有效砌块
        elif state[2] == self.padding_idx:
            #如果第一个反应为单组分反应，返回单组分反应产物
            if state[3] in self.uni_list:
                return self.get_uni_product(self.inverse_lookup_2[state[0]],self.inverse_lookup_1[state[3]],state[4])
            else:
                return self.get_bi_product(self.inverse_lookup_2[state[0]],self.inverse_lookup_2[state[1]],self.inverse_lookup_1[state[3]],state[4])
        #三个有效砌块
        else:
            #先判断中间产物是单组分还是双组分反应产物
            if state[3] in self.uni_list:
                intermediate_smiles = self.get_uni_product(self.inverse_lookup_2[state[0]],self.inverse_lookup_1[state[3]],state[4])
            else:
                intermediate_smiles = self.get_bi_product(self.inverse_lookup_2[state[0]],self.inverse_lookup_2[state[1]],self.inverse_lookup_1[state[3]],state[4])
            #再判断终产物是单组分还是双组分反应产物
            if state[5] in self.uni_list:
                final_smiles = self.get_uni_product(intermediate_smiles,self.inverse_lookup_1[state[5]],state[6])
            else:
                final_smiles = self.get_bi_product(intermediate_smiles,self.inverse_lookup_2[state[2]],self.inverse_lookup_1[state[5]],state[6])
            return final_smiles
        
    def statetorch2readable(self, state: TensorType["1", "max_seq_length"]) -> str:
        #获取state的有效长度
        if state[0][-1].item() == self.padding_idx:
            state = state[: torch.where(state == self.padding_idx)[0][0]]
        state = state[0].tolist()
        return self.state2readable(state)

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

    #当前写法针对state的前三个砌块进行考虑，输入的action_1和action_2并无意义，完全可以根据state返回父节点和父动作
    def get_parents(self, state=None, done=None):
        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [state], [(self.eos_1,)], [(self.eos_2,)]
        elif torch.eq(state, self.source).all():#如果state的每个元素都是初始元素
            return [], [], []
        else:
            parents = []
            actions_1 = []
            actions_2 = []
            if state[1] == self.padding_idx:#如果输入state只有一个砌块，其父节点为初始节点，动作1为空，动作2为选择的该砌块
                parent_action_2 = tuple (state[0:1].numpy())
                actions_2.append(parent_action_2)
                return [self.source],actions_1,actions_2
            elif state[2] == self.padding_idx:#如果输入state只有两个砌块
                parent_actions_1 = tuple(state[3:4].numpy())
                actions_1.append(parent_actions_1)
                
                parent_action_2 = tuple(state[1:2].numpy())
                actions_2.append(parent_action_2)
                
                parent = state.clone().detach()
                parent[1] = self.padding_idx
                parents.append(parent)
                return parents,actions_1,actions_2
            else:#如果输入state有三个砌块
                parent_actions_1 = tuple(state[5:6].numpy())
                actions_1.append(parent_actions_1)
                
                parent_action_2 = tuple(state[2:3].numpy())
                actions_2.append(parent_action_2)
                
                parent = state.clone().detach()
                parent[2] = self.padding_idx
                parents.append(parent)
                return parents,actions_1,actions_2

    #基于输入的动作，返回下一个state
    #动作1选模板，动作2选砌块
    def step(self, action_1: Tuple[int],action_2: Tuple[int]) -> Tuple[List[int], Tuple[int, int], Tuple[int, int], bool]:
        assert action_1 in self.action_space_1
        assert action_2 in self.action_space_2
        
        #如果state（数组形式）的最后一个索引不是填充索引，说明当前state是最终状态
        if self.state[-1] != self.padding_idx:
            self.done = True
            self.n_actions += 1
            return self.state, (self.eos_1,), (self.eos_2,), True#valid=True表示当前动作可用
        #如果state为初始state，不考虑action_1
        if self.state[0] == self.padding_idx:
            state_next = self.state.clone().detach()
            state_next[0] = torch.LongTensor(action_2[0])
            valid = True
            self.state = state_next
            self.n_actions += 1
            return self.state, None, action_2, valid
        
        #如果输入的动作1和动作2都不是eos
        if action_1[0] != self.eos_1 and action_2[0] != self.eos_2:
            #1.如果当前state[1]为pad，基于action_1和action_2对state[1]和state[3],state[4]进行更新
            if self.state[1] == self.padding_idx:
                state_next = self.state.clone().detach()
                #1.选择的action1为单组分反应
                if action_1[0] in self.uni_list:
                    state_next[1] = torch.LongTensor(action_2[0])
                    state_next[3] = torch.LongTensor(action_1[0])
                    state_next[4] = torch.LongTensor(self.get_uni_product_idx(self.state2readable(state_next),action_1[0]))
                    valid = True
                    self.state = state_next
                    self.n_actions += 1
                    return self.state, action_1, action_2, valid
                #2.选择的action1为双组分反应
                else:
                    state_next[1] = torch.LongTensor(action_2[0])
                    state_next[3] = torch.LongTensor(action_1[0])
                    state_next[4] = torch.LongTensor(self.get_bi_product_idx(self.state2readable(state_next),action_1[0],action_2[0]))
                    valid = True
                    self.state = state_next
                    self.n_actions += 1
                    return self.state, action_1, action_2, valid
            #2.如果当前state[2]为pad，基于action_1和action_2对state[2]和state[5]，state[6]进行更新
            elif self.state[2] == self.padding_idx:
                state_next = self.state.clone().detach()
                #1.选择的action1为单组分反应
                if action_1[0] in self.uni_list:
                    state_next[2] = torch.LongTensor(action_2[0])
                    state_next[5] = torch.LongTensor(action_1[0])
                    state_next[6] = torch.LongTensor(self.get_uni_product_idx(self.state2readable(state_next),action_1[0]))
                    valid = True
                    self.state = state_next
                    self.n_actions += 1
                    return self.state, action_1, action_2, valid
                #2.选择的action2为双组分反应   
                else:
                    state_next[2] = torch.LongTensor(action_2[0])
                    state_next[5] = torch.LongTensor(action_1[0])
                    state_next[6] = torch.LongTensor(self.get_bi_product_idx(self.state2readable(state_next),action_1[0],action_2[0]))
                    valid = True
                    self.state = state_next
                    self.n_actions += 1
                    return self.state, action_1, action_2, valid
        #输入的动作1和动作2中存在一个eos
        else:
            self.done = True
            valid = True
            self.n_actions += 1
            return self.state, action_1, action_2, valid
