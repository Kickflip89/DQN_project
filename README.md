# DQN_project
Deep Reinforcement Learning Project

## Model Initialization
Use one of the following to intialize a network (no args right now, have to modify code)
```python
from networks.DQNetwork import LearningNetwork #DQN implementation
from networks.DDQNetwork import DLearningNetwork #DDQN implementation
from networks.SQNetwork import SLearningNetwork #SQN implementation
from networks.DSQNetwork import DSLearningNetwork #DSQN implementation
```

To use the modified reward system in DQN or DDQN, uncomment the line:
```python
#total_reward -= 10
```
in the q_iteration method of the LearningNetwork class.

## Training:
To train a model, modify train.py or write your own script.

## Playing:
Most networks have a .load() method to load a trained model and a .play()
method which uses matplotlib vice opengl to render frames.  play.py contains
a sample driver for a DSLearningNetwork

## Data bug for iterations
in ./data, the lists representing the average number of iterations for dqn and ddqn were not reset during training due to a bug.  Thankfully, since they were
calculated at regular intervals, the actual average iterations can be calculated by subtracting the current element from the previous element. The true values can be found with

```python
#its is of form {model_name:data} and has the bugged data for average iterations
for model, data in its.items():
    new_dat = [data[0]]
    for i in range(1,len(data)):
        new_dat.append(data[i] - data[i-1])
    its[model] = new_dat
```
