from typing import List
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import json
import gzip
import time
with gzip.GzipFile('/home/lishuwang/project/GFlowNet/lp_gflownet/gflownet/data/smiles_list.json.gz','r') as f_in:
    smiles_list = json.load(f_in)
from gflownet.envs.sequence import Sequence
from gflownet.data.template import smarts_list, name_list 

#功能：输出训练集（从oracle.initializeDataset）;挑选测试集（从env中获取）
class Aptamers(Sequence):
    def __init__(
        self,
        **kwargs,
    ):
        special_tokens_1 = ["EOS"]
        special_tokens_2 = ["EOS","UNI","PAD"]
        self.vocab_1 = smarts_list + special_tokens_1
        self.vocab_2 = smiles_list + special_tokens_2
        super().__init__(
            **kwargs,
            special_tokens_1=special_tokens_1,special_tokens_2=special_tokens_2,
        )

        if (
            hasattr(self, "proxy")
            and self.proxy is not None
            and hasattr(self.proxy, "setup")#如果proxy不为空且proxy有setup属性（即proxy是一个类）
        ):
            self.proxy.setup(self.max_seq_length)#env导入proxy，用于计算每个state的y值
    
        
    #构建一个*随机采样的训练集（用于gflownet的训练集，以降低loss）
    #输出train_df（csv格式，可指定输出位置），包含三列（samples,indices,energies）
        #samples_dict = oracle.initializeDataset和state_letters = oracle.numbers2letters(samples_mat)这两行需要修改
    def make_train_set(
        self,
        ntrain,
        oracle=None,
        seed=168,#挑选训练集的随机种子
        output_csv=None,
    ):
        """
        Constructs a randomly sampled train set.
        
        output_csv: str存储训练集的路径地址
            Optional path to store the test set as CSV.
        """
        samples_dict = oracle.initializeDataset(
            save=False, returnData=True, customSize=ntrain, custom_seed=seed
        )
        energies = samples_dict["energies"]
        samples_mat = samples_dict["samples"]#[[0,1,2,3],[0,1,3,2]]
        state_letters = oracle.numbers2letters(samples_mat)#['smiles1_tag1,smiesl2_tag2,smiles3_tag3']
        # state_ints = [
        #     "".join([str(el) for el in state if el > 0]) for state in samples_mat
        # ]#[[0,1,2,3,],[0,1,3,2]]→['123','132']这里有点问题，索引为0的元素被忽略了，这里的索引0应该也是指代的一个smiles_tag
        state_ints = [
            "".join([str(el) for el in state if el >= 0]) for state in samples_mat
        ]
        if isinstance(energies, dict):
            energies.update({"samples": state_letters, "indices": state_ints})#添加两个新的键值对，samples对应碱基序列型的样本，indices对应其数字字符串形式
            df_train = pd.DataFrame(energies)#dict转换为df，df_train包含['samples'],['indices'],['energies']三列
        else:
            df_train = pd.DataFrame(
                {"samples": state_letters, "indices": state_ints, "energies": energies}
            )
        if output_csv:
            df_train.to_csv(output_csv)#将训练集保存为csv文件并存储
        return df_train

    #从base环境下挑选样本作为测试集（这里的挑选思路是以y值为范围，按均匀分布进行采样）
    def make_test_set(
        self,
        path_base_dataset,
        ntest,
        min_length=0,
        max_length=np.inf,
        seed=167,#挑选测试集的种子数
        output_csv=None,
    ):
        """
        Constructs an approximately uniformly distributed (on the score) set, by
        selecting samples from a larger base set.
        #从一个较大的基础集合中选择样本，构建一个近似均匀分布（在分数上）的集合。

        Args
        ----
        path_base_dataset : str
            Path to a CSV file containing the base data set.
        #base dataset的位置，base dataset和训练集一样，df有三列
        
        ntest : int
            Number of test samples.

        seed : int
            Random seed.

        output_csv: str
            Optional path to store the test set as CSV.
        """
        if path_base_dataset is None:
            return None, None
        times = {
            "all": 0.0,
            "indices": 0.0,
        }#记录时间，all表示总时间，indices表示挑选索引的时间
        t0_all = time.time()#记录开始时间
        if seed:
            np.random.seed(seed)
        df_base = pd.read_csv(path_base_dataset, index_col=0)#读取base dataset，index_col=0表示第一列作为索引
        df_base = df_base.loc[
            (df_base["samples"].map(len) >= min_length)
            & (df_base["samples"].map(len) <= max_length)
        ]#只挑选长度在min_length和max_length之间的样本（其实没太大必要）
        energies_base = df_base["energies"].values#array([值1，值2，值3，...])
        min_base = energies_base.min()#最小能量
        max_base = energies_base.max()#最大能量
        distr_unif = np.random.uniform(low=min_base, high=max_base, size=ntest)#array([值1，值2，值3，...]])，这里的值排列顺序是随机的，并且这里的值不一定在实际能量值中存在

        t0_indices = time.time()#记录开始挑选索引的时间
        idx_samples = []
        #这里挑选测试集的思路值得学习，distr_unif是均匀采样的能量值列表，最终选取的测试集中的每个样本都分别对应和distr_unif中能量值最接近的样本
        for idx in tqdm(range(ntest)):#tqdm是一个进度条
            dist = np.abs(energies_base - distr_unif[idx])#energies_base这个array中的每个元素都减去distr_unif[idx]对应的能量值，再取绝对值
            idx_min = np.argmin(dist)#返回最小值（能量值和随机选的数的能量值的差值）的索引
            if idx_min in idx_samples:
                idx_sort = np.argsort(dist)#返回dist中元素从小到大排序后的索引
                for idx_next in idx_sort:
                    if idx_next not in idx_samples:
                        idx_samples.append(idx_next)
                        break
            else:
                idx_samples.append(idx_min)
        t1_indices = time.time()#记录结束挑选索引的时间
        times["indices"] += t1_indices - t0_indices#记录挑选索引的时间
        # Make test set
        df_test = df_base.iloc[idx_samples]#根据索引挑选样本
        if output_csv:
            df_test.to_csv(output_csv)
        t1_all = time.time()#记录结束时间
        times["all"] += t1_all - t0_all#记录总时间
        return df_test, times
