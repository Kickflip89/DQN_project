# DQN_project
Deep Reinforcement Learning Project

The repository is a little sloppy and not well documented at this point

##Model Initialization
Use one of the following to intialize a network (no args right now, have to modify code)
```python
from DQNetwork import LearningSystem #DQN implementation
from DDQNetwork import DLearningSystem #DDQN implementation
from SQNetwork import SLearningSystem #SQN implementation
```

To use the modified reward system in DQN or DDQN, uncomment the line:
```python
#reward -= 10
```
in the q_iteration method of the LearningSystem class.
