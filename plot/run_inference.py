#
# Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set_theme()

df = pd.read_csv("../result/inference.csv")
print(df)
width = 0.5

figure, ax = plt.subplots(1, 3)
bottom = np.zeros(4)

tmp=df[(df.algo==0) & (df.v==64)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
print(tmp)
ax[0].set_title("BERT-large, bs=32")
tmp.plot.bar(x='m',y='mean', ax=ax[0])

tmp=df[(df.algo==1) & (df.v==64)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
ax[1].set_title("GPT2-large, bs=8")
tmp.plot.bar(x='m',y='mean', ax=ax[1])

tmp=df[(df.algo==2) & (df.v==64)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
ax[2].set_title("GPT3, bs=1")
tmp.plot.bar(x='m',y='mean', ax=ax[2])

#ax.set_title("Number of penguins with above average body mass")
figure.tight_layout()
ax[1].set_xlabel("Sparsity", fontsize=14)
ax[0].set_ylabel("Latency(ms)", fontsize=14)
plt.savefig('inference_v64.png')


##############
figure, ax = plt.subplots(1, 3)
bottom = np.zeros(4)

tmp=df[(df.algo==0) & (df.v==128)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
print(tmp)
ax[0].set_title("BERT-large, bs=32")
tmp.plot.bar(x='m',y='mean', ax=ax[0])

tmp=df[(df.algo==1) & (df.v==128)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
ax[1].set_title("GPT2-large, bs=8")
tmp.plot.bar(x='m',y='mean', ax=ax[1])

tmp=df[(df.algo==2) & (df.v==128)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
ax[2].set_title("GPT3, bs=1")
tmp.plot.bar(x='m',y='mean', ax=ax[2])

#ax.set_title("Number of penguins with above average body mass")
figure.tight_layout()
ax[1].set_xlabel("Sparsity", fontsize=14)
ax[0].set_ylabel("Latency(ms)", fontsize=14)
plt.savefig('inference_v128.png')