#import agent of choice
#from networks.SQNetwork import SLearningNetwork
from networks.DSQNetwork import DSLearningNetwork
#from netwroks.DDQNetwork import DLearningNetwork
#from networks.DQNetwork import LearningNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
  #example prolong_life agent
  network = DSLearningNetwork(lam_r=.5, a_r=.5)
  network.load()
  network.play()
