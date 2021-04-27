#import agent of choice
from SQNetwork import SLearningNetwork
#from DDQNetwork import DLearningNetwork
#from DQNetwork import LearningNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == '__main__':
  #example prolong_life agent
  network = SLearningNetwork(lam_p=10, a_r=.7)
  network.train(50000)
  
  scores = network.score_history
  its = network.its_hist
  x = range(len(its))
  
  sns.set()
  fig, ax1 = plt.subplots()
  lin1 = ax1.plot(x, scores, label='Score')
  ax1.set_xlabel('Epoch x 100')
  ax1.set_ylabel('Avg SPE')
  ax2 = ax1.twinx()
  ax2.plot([0],[0])
  lin2 = ax2.plot(x, its, label='Iterations')
  ax2.set_ylabel('Avg APE')
  plt.title('Average APE and SPE in 100 Epoch Intervals')
  lins = lin1 + lin2
  lbls = [l.label() for l in lins]
  ax1.legend(lins, lbls)
  plt.show()
