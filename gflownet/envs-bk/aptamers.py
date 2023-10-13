"""
Classes to represent aptamers environments
"""
from typing import List
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import time
from gflownet.utils.sequence.aptamers import NUCLEOTIDES#四个碱基的列表（碱基为str）
from gflownet.envs.sequence import Sequence

#功能：输出训练集（从oracle.initializeDataset）;挑选测试集（从env中获取）
class Aptamers(Sequence):
    """
    Aptamer sequence environment
    """

    def __init__(
        self,
        **kwargs,
    ):
        special_tokens = ["PAD", "EOS"]#特殊标记,pad表示开始填充，eos表示序列结束
        self.vocab = NUCLEOTIDES + special_tokens#这里可以将NUCLEOTIDES替换为bricks（以smiles形式表示每个brick）
        super().__init__(
            **kwargs,
            special_tokens=special_tokens,
        )

        if (
            hasattr(self, "proxy")
            and self.proxy is not None
            and hasattr(self.proxy, "setup")
        ):
            self.proxy.setup(self.max_seq_length)#env导入proxy，用于计算每个state的y值

    #构建一个随机采样的训练集（用于gflownet的训练集，以降低loss）
    #输出train_df，包含三列（samples,indices,energies），samples即碱基序列
    def make_train_set(
        self,
        ntrain,
        oracle=None,
        seed=168,
        output_csv=None,
    ):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        ntest : int
            Number of test samples.

        seed : int
            Random seed.

        output_csv: str
            Optional path to store the test set as CSV.
        """
        samples_dict = oracle.initializeDataset(
            save=False, returnData=True, customSize=ntrain, custom_seed=seed
        )#返回一个字典，包含了ntrain个样本的序列和能量（这个dict同样是oracle需要的dataset形式）
        energies = samples_dict["energies"]#样本对应的能量
        samples_mat = samples_dict["samples"]#样本（即states，每个states以数字形式存储）
        state_letters = oracle.numbers2letters(samples_mat)#（将数字形式的样本转换为碱基字母形式）
        state_ints = [
            "".join([str(el) for el in state if el > 0]) for state in samples_mat
        ]#将数字形式的states转换为字符串形式（相当于是一个列表，每个元素是state的字符串形式）
        if isinstance(energies, dict):
            energies.update({"samples": state_letters, "indices": state_ints})#添加两个新的键值对，samples对应碱基序列型的样本，indices对应其数字字符串形式
            df_train = pd.DataFrame(energies)#将字典转换为dataframe
        else:#如果energies不是字典形式
            df_train = pd.DataFrame(
                {"samples": state_letters, "indices": state_ints, "energies": energies}
            )
        if output_csv:
            df_train.to_csv(output_csv)#将训练集保存为csv文件并存储
        return df_train

    # TODO: improve approximation of uniform
    #从base环境下挑选样本作为测试集（这里的挑选思路是以y值为范围，按均匀分布进行采样）
    def make_test_set(
        self,
        path_base_dataset,
        ntest,
        min_length=0,
        max_length=np.inf,
        seed=167,#注意seed数和make_train_set中的seed数不同
        output_csv=None,
    ):
        """
        Constructs an approximately uniformly distributed (on the score) set, by
        selecting samples from a larger base set.#从一个较大的基础集合中选择样本，构建一个近似均匀分布（在分数上）的集合。

        Args
        ----
        path_base_dataset : str
            Path to a CSV file containing the base data set.

        ntest : int
            Number of test samples.

        seed : int
            Random seed.

        dask : bool
            If True, use dask to efficiently read a large base file.#如果为True，则使用dask来高效读取大型基本文件。

        output_csv: str
            Optional path to store the test set as CSV.
        """
        if path_base_dataset is None:
            return None, None
        times = {
            "all": 0.0,
            "indices": 0.0,
        }
        t0_all = time.time()
        if seed:
            np.random.seed(seed)
        df_base = pd.read_csv(path_base_dataset, index_col=0)
        df_base = df_base.loc[
            (df_base["samples"].map(len) >= min_length)
            & (df_base["samples"].map(len) <= max_length)
        ]
        energies_base = df_base["energies"].values
        min_base = energies_base.min()#最小能量
        max_base = energies_base.max()#最大能量
        distr_unif = np.random.uniform(low=min_base, high=max_base, size=ntest)#从min_base到max_base中随机选择ntest个数
        # Get minimum distance samples without duplicates#获取没有重复的最小距离样本
        t0_indices = time.time()
        idx_samples = []
        for idx in tqdm(range(ntest)):
            dist = np.abs(energies_base - distr_unif[idx])
            idx_min = np.argmin(dist)#返回最小值（能量值和随机选的数的能量值的差值）的索引
            if idx_min in idx_samples:
                idx_sort = np.argsort(dist)#返回从小到大排序的索引
                for idx_next in idx_sort:
                    if idx_next not in idx_samples:
                        idx_samples.append(idx_next)
                        break
            else:
                idx_samples.append(idx_min)
        t1_indices = time.time()
        times["indices"] += t1_indices - t0_indices
        # Make test set
        df_test = df_base.iloc[idx_samples]
        if output_csv:
            df_test.to_csv(output_csv)
        t1_all = time.time()
        times["all"] += t1_all - t0_all
        return df_test, times
