https://arxiv.org/pdf/2007.05156.pdf    
**https://arxiv.org/ftp/arxiv/papers/2010/2010.06187.pdf** 


**different sensor input information**

try sparse in road      
conv in channel direction       
_____________________________________________________________________

2.2.2.4 Imitation learning and inverse RL
Imitation learning is a process of learning from demonstrations which is also known as 
“apprenticeship learning”. The idea is what if the agent has no idea what the reward is, how the agent 
can approximate its behavior to find the best policy without any reward to guide. In this case, with a 
set of expert demonstrations (typically defined by humans), the agent tries to learn the optimal policy 
imitating the expert’s decisions. Usually, the expert demonstrations are provided in the form of some
trajectories (𝜏 = 𝑠0, 𝑎0,𝑠1, 𝑎1, ….), where these trajectories are built on some good policies. Depending
on the loss function and the learning algorithm, there can be variations in the approach of imitation 
learning. One way to learn from the expert demonstration is to extract the reward signal which is known 
as Inverse RL (Ng and Russell, 2000). In inverse RL, the agent first learns a reward signal from the 
expert demonstrations, and then uses this reward function to find the optimal policy. Through inverse 
RL, sometimes it is difficult to extract a unique reward signal, multiple reward signals can result in the 
same reward value for a given policy. However, it is possible to find several candidate reward signals. 
With some strong assumptions, including thorough knowledge of the environment dynamics and 
completely solving the problem several times, a proper reward signal can be extracted to find the 
optimal policy (Sutton and Barto, 2018). Inverse RL can be applied in a DRL framework by
incorporated deep neural networks as a function approximator. For more comprehensive readings on 
imitation learning and inverse RL in the DRL framework, we recommend readers the following 
literature (Stadie et al., 2017; Hester et al., 2018; Wulfmeier et al., 2015).
In many real-world problems, it is very difficult to derive a proper reward function. The successful 
completion of a task may depend on multiple factors that require some parameterization of the reward 
function. Weights of these parameters need to be defined manually which involves an intense tuning 
process. For instance, the success of lane change not only depends on whether the agent successfully 
completes the task but also depends on the marginal safety risk, smoothness, and comfortability of the 
transition. It is difficult to fix the weight for each of these factors in the reward function beforehand 
without knowing the marginal effect of any changes in the weight on the agent’s behavior. Similarly, 
any task that requires to optimize multiple factors through a common reward signal can utilize inverse 
RL to ease the overall learning efficiency.
