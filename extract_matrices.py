import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

set1 = np.load('set1.npy',encoding='bytes')
set1 = set1.item()
pctm = np.mean(set1[b'pitch_class_transition_matrix'],axis=0)
sns.heatmap(pctm)
plt.show()
