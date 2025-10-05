[![logo](https://services.dev.arxiv.org/html/static/arxiv-logomark-small-white.svg)Back to arXiv](https://arxiv.org/)

[Back to abstract page](https://arxiv.org/abs/2510.01051v1)

[![logo](https://services.dev.arxiv.org/html/static/arxiv-logo-one-color-white.svg)Back to arXiv](https://arxiv.org/)

This is **experimental HTML** to improve accessibility. We invite you to report rendering errors. Use Alt+Y to toggle on accessible reporting links and Alt+Shift+Y to toggle off. Learn more [about this project](https://info.arxiv.org/about/accessible_HTML.html) and [help improve conversions](https://info.arxiv.org/help/submit_latex_best_practices.html).

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) [Report Issue](https://arxiv.org/html/2510.01051v1/#myForm) [Back to Abstract](https://arxiv.org/abs/2510.01051v1) [Download PDF](https://arxiv.org/pdf/2510.01051v1)

## Table of Contents

01. [Abstract](https://arxiv.org/html/2510.01051v1#abstract "Abstract")
02. [1 Introduction](https://arxiv.org/html/2510.01051v1#S1 "In GEM: A Gym for Agentic LLMs")
03. [2 GEM environments](https://arxiv.org/html/2510.01051v1#S2 "In GEM: A Gym for Agentic LLMs")    1. [2.1 Interface](https://arxiv.org/html/2510.01051v1#S2.SS1 "In 2 GEM environments ‣ GEM: A Gym for Agentic LLMs")
    2. [2.2 Tasks and tools](https://arxiv.org/html/2510.01051v1#S2.SS2 "In 2 GEM environments ‣ GEM: A Gym for Agentic LLMs")
    3. [2.3 Asynchronous vectorization and autoreset](https://arxiv.org/html/2510.01051v1#S2.SS3 "In 2 GEM environments ‣ GEM: A Gym for Agentic LLMs")
    4. [2.4 Wrappers](https://arxiv.org/html/2510.01051v1#S2.SS4 "In 2 GEM environments ‣ GEM: A Gym for Agentic LLMs")
04. [3 Reinforcement learning with GEM](https://arxiv.org/html/2510.01051v1#S3 "In GEM: A Gym for Agentic LLMs")    1. [3.1 Preliminary: LLMs as agents](https://arxiv.org/html/2510.01051v1#S3.SS1 "In 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs")
    2. [3.2 Baseline algorithms](https://arxiv.org/html/2510.01051v1#S3.SS2 "In 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs")
05. [4 Empirical studies with GEM](https://arxiv.org/html/2510.01051v1#S4 "In GEM: A Gym for Agentic LLMs")    1. [4.1 Benchmarking RL algorithms for LLMs](https://arxiv.org/html/2510.01051v1#S4.SS1 "In 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")
    2. [4.2 Discount factor γ\\gamma matters](https://arxiv.org/html/2510.01051v1#S4.SS2 "In 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")
    3. [4.3 Tool-integration in math and question-answering tasks](https://arxiv.org/html/2510.01051v1#S4.SS3 "In 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")
    4. [4.4 Studying generalization](https://arxiv.org/html/2510.01051v1#S4.SS4 "In 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")
    5. [4.5 Integration with training frameworks](https://arxiv.org/html/2510.01051v1#S4.SS5 "In 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")
06. [5 Agent evaluation with GEM](https://arxiv.org/html/2510.01051v1#S5 "In GEM: A Gym for Agentic LLMs")    1. [5.1 General tool use via model context protocol](https://arxiv.org/html/2510.01051v1#S5.SS1 "In 5 Agent evaluation with GEM ‣ GEM: A Gym for Agentic LLMs")
    2. [5.2 Terminal environment via Docker container](https://arxiv.org/html/2510.01051v1#S5.SS2 "In 5 Agent evaluation with GEM ‣ GEM: A Gym for Agentic LLMs")
07. [6 Conclusions](https://arxiv.org/html/2510.01051v1#S6 "In GEM: A Gym for Agentic LLMs")
08. [A Environment registration](https://arxiv.org/html/2510.01051v1#A1 "In GEM: A Gym for Agentic LLMs")
09. [B Case studies of language games](https://arxiv.org/html/2510.01051v1#A2 "In GEM: A Gym for Agentic LLMs")
10. [C Algorithm](https://arxiv.org/html/2510.01051v1#A3 "In GEM: A Gym for Agentic LLMs")
11. [D Extended empirical studies with GEM](https://arxiv.org/html/2510.01051v1#A4 "In GEM: A Gym for Agentic LLMs")    1. [D.1 Improving learning efficiency via return batch normalization (ReBN)](https://arxiv.org/html/2510.01051v1#A4.SS1 "In Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")
    2. [D.2 RL on vision-language environments](https://arxiv.org/html/2510.01051v1#A4.SS2 "In Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")
    3. [D.3 Multi-agent environments](https://arxiv.org/html/2510.01051v1#A4.SS3 "In Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")
12. [E Related works](https://arxiv.org/html/2510.01051v1#A5 "In GEM: A Gym for Agentic LLMs")
13. [F Experimental settings](https://arxiv.org/html/2510.01051v1#A6 "In GEM: A Gym for Agentic LLMs")
14. [References](https://arxiv.org/html/2510.01051v1#bib "References")

[License: CC BY 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2510.01051v1 \[cs.LG\] 01 Oct 2025

# GEM: A Gym for Agentic LLMs

Report issue for preceding element

Zichen Liu†12, Anya Sims†13\*, Keyu Duan†12\*, Changyu Chen†14\*, Simon Yu69, Xiangxin Zhou1\*

Haotian Xu7, Shaopan Xiong8, Bo Liu2, Chenmien Tan9, Chuen Yang Beh2, Weixun Wang8

Hao Zhu5, Weiyan Shi6, Diyi Yang5, Michael Shieh2, Yee Whye Teh3, Wee Sun Lee2, Min Lin1

1Sea AI Lab 2NUS
3Oxford
4SMU
5Stanford
6Northeastern
7OpenRLHF
8ROLL
9RL2
†Equal contribution with random order. ∗Work done during their associate membership at Sea AI Lab.

Report issue for preceding element

## Abstract

Report issue for preceding element

The training paradigm for large language models (LLMs) is moving from static datasets to experience-based learning, where agents acquire skills via interacting with complex environments. To facilitate this transition we introduce GEM (General Experience Maker), an open-source environment simulator designed for the age of LLMs. Analogous to OpenAI-Gym for traditional reinforcement learning (RL), GEM provides a standardized framework for the environment-agent interface, including asynchronous vectorized execution for high throughput, and flexible wrappers for easy extensibility. GEM also features a diverse suite of environments, robust integrated tools, and single-file example scripts demonstrating using GEM with five popular RL training frameworks. Along with this, we also provide a set of baselines across 24 environments using REINFORCE with Return Batch Normalization (ReBN), which—unlike GRPO—is compatible with the full RL setting of dense per-turn rewards and offers better credit assignment. We further conduct apple-to-apple benchmarking of PPO, GRPO and REINFORCE in both single- and multi-turn settings using GEM to shed light on the algorithmic designs. Lastly, GEM also functions as a convenient evaluation toolkit besides a training environment. We hope this framework can help accelerate future agentic LLM research111Code is available at: [https://github.com/axon-rl/gem](https://github.com/axon-rl/gem "")..

Report issue for preceding element

![Refer to caption](https://arxiv.org/html/2510.01051v1/x1.png)Figure 1: Learning curves of Qwen3-based agents across diverse environments of 5 categories: game (language games); rg (ReasoningGym); code (coding tasks); math (python-integrated math questions); qa (search-integrated general questions). All agents are learned via a simple yet general multi-turn algorithm based on REINFORCE ( [Algorithm˜1](https://arxiv.org/html/2510.01051v1#alg1 "In Appendix C Algorithm ‣ GEM: A Gym for Agentic LLMs")). The comparison between two curves in each subplot illustrate the effectiveness of Return Batch Normalization (ReBN).Report issue for preceding element

## 1 Introduction

Report issue for preceding element

Reinforcement learning (RL) (Sutton and Barto, [2018](https://arxiv.org/html/2510.01051v1#bib.bib38 "")) has emerged as a powerful paradigm for improving the reasoning capabilities of large language models (LLMs) (OpenAI, [2024](https://arxiv.org/html/2510.01051v1#bib.bib26 ""); Guo et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib15 "")). By collecting experience in interactive environments, RL allows agents to learn complex, open-ended tasks without direct supervision (Silver and Sutton, [2025](https://arxiv.org/html/2510.01051v1#bib.bib35 "")).
This approach promises to create powerful agents for a variety of domains. For instance, an agent could develop entire software modules by writing, testing, and debugging code, while also adapting to integration failures or evolving requirements. Similarly, in scientific discovery, an agent could be trained to develop hypotheses, design relevant experiments, and adjust its long-term strategy based on the results.

Report issue for preceding element

However, current research on RL for LLMs has largely focused on single-turn tasks, such as answering math questions or retrieving specific data (Lambert et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib22 ""); Guo et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib15 "")). While these tasks are a valuable starting point, they significantly oversimplify multi-turn interactions (Liu et al., [2025a](https://arxiv.org/html/2510.01051v1#bib.bib23 "")). This oversimplification means that algorithms which excel in the single-turn setting (e.g., GRPO (Shao et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib33 ""))) are fundamentally inapplicable to full multi-turn problems. If the goal is to train agentic LLMs capable of long-horizon planning, trial-and-error, iterative refinement etc, it is crucial to transition to testbeds that support these more complex multi-turn interactions.

Report issue for preceding element

To facilitate this next step, we introduce GEM (General Experience Maker), an open-source environment framework for diverse, multi-turn, long-horizon tasks. Motivated by OpenAI-Gym (Brockman et al., [2016](https://arxiv.org/html/2510.01051v1#bib.bib7 "")) which catalyzed research in traditional RL by providing a unified interface and standardized environments, GEM aims to provide analogous foundational infrastructure for LLM agents. GEM offers a diverse suite of environments spanning single- and multi-turn (over 100100 turns) tasks (including tool integrated responses, reasoning games etc), flexible observation and action wrappers, asynchronous parallel execution, and a rich set of tools (python, search, and external MCP compatible tools). Additionally, GEM includes validated baselines and single-file training scripts showcasing seamless integration with five popular RL training frameworks (Oat, Verl, OpenRLHF, ROLL, and RL2—see [Section˜4.5](https://arxiv.org/html/2510.01051v1#S4.SS5 "4.5 Integration with training frameworks ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")).

Report issue for preceding element

Besides introducing the GEM framework, this paper also presents and discusses a simple yet effective algorithmic variant of REINFORCE (Williams, [1992](https://arxiv.org/html/2510.01051v1#bib.bib45 "")) which incorporates Return Batch Normalization (ReBN), a useful technique similar to advantage normalization (Andrychowicz et al., [2021](https://arxiv.org/html/2510.01051v1#bib.bib3 ""); Liu et al., [2025b](https://arxiv.org/html/2510.01051v1#bib.bib25 "")) that brings consistent improvements ( [Figure˜1](https://arxiv.org/html/2510.01051v1#S0.F1 "In GEM: A Gym for Agentic LLMs")).
Unlike GRPO and its variants, REINFORCE with ReBN is fully compatible with the multi-turn RL setting, including turn-level dense rewards and arbitrary discount factors.
We further compare REINFORCE-based algorithms with multi-turn PPO (Schulman et al., [2017](https://arxiv.org/html/2510.01051v1#bib.bib32 "")) and GRPO, showing its theoretical connections and empirical tradeoffs.
We also provide case studies on the impact of the discount factor γ\\gamma on multi-turn learning, extensive results of tool-integrated RL, and performance benchmarks on terminal and MCP usage of strong LLMs using GEM as a unified evaluation toolkit. We hope this framework will accelerate RL research on agentic LLMs and advance progress toward more capable and autonomous AI systems.

Report issue for preceding element

## 2 GEM environments

Report issue for preceding element

This section introduces GEM’s core functionality, covering its main interface ( [Section˜2.1](https://arxiv.org/html/2510.01051v1#S2.SS1 "2.1 Interface ‣ 2 GEM environments ‣ GEM: A Gym for Agentic LLMs")), the environment design ( [Section˜2.2](https://arxiv.org/html/2510.01051v1#S2.SS2 "2.2 Tasks and tools ‣ 2 GEM environments ‣ GEM: A Gym for Agentic LLMs")), and advanced features such as asynchronous vectorization and modular wrappers ( [Sections˜2.3](https://arxiv.org/html/2510.01051v1#S2.SS3 "2.3 Asynchronous vectorization and autoreset ‣ 2 GEM environments ‣ GEM: A Gym for Agentic LLMs") and [2.4](https://arxiv.org/html/2510.01051v1#S2.SS4 "2.4 Wrappers ‣ 2 GEM environments ‣ GEM: A Gym for Agentic LLMs")).

Report issue for preceding element

### 2.1 Interface

Report issue for preceding element

GEM employs a standardized environment interface closely following the well-established OpenAI Gym API with the main functions being reset() and step().
A basic agent-environment interaction loop is as follows (multi-agent interface shown in [Section˜D.3](https://arxiv.org/html/2510.01051v1#A4.SS3 "D.3 Multi-agent environments ‣ Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")):

Report issue for preceding element

[⬇](data:text/plain;base64,aW1wb3J0IGdlbQojIGdlbS5wcmludF9lbnZzKCkgIyB0byBsaXN0IGFsbCBhdmFpbGFibGUgZW52aXJvbm1lbnRzCmVudiA9IGdlbS5tYWtlKCJnYW1lOkd1ZXNzVGhlTnVtYmVyLXYwIikKb2JzZXJ2YXRpb24sIGluZm8gPSBlbnYucmVzZXQoKQoKd2hpbGUgVHJ1ZToKICAgICMgKDEpIEFnZW50IGFjdGluZzoKICAgIGFjdGlvbiA9IGVudi5zYW1wbGVfcmFuZG9tX2FjdGlvbigpCiAgICAjIGFjdGlvbiA9IGFnZW50LmFjdChvYnNlcnZhdGlvbikgIyByZWFsIGFjdGluZyBieSBMTE0gc2FtcGxpbmcKCiAgICAjICgyKSBFbnZpcm9ubWVudCBleGVjdXRpb246CiAgICBuZXh0X29icywgcmV3YXJkLCB0ZXJtaW5hdGVkLCB0cnVuY2F0ZWQsIGluZm8gPSBlbnYuc3RlcChhY3Rpb24pCgogICAgIyAoMykgQWdlbnQgbGVhcm5pbmc6CiAgICAjIGFnZW50LmxlYXJuKG9ic2VydmF0aW9uLCBhY3Rpb24sIHJld2FyZCkKCiAgICBvYnNlcnZhdGlvbiA9IG5leHRfb2JzCiAgICBpZiB0ZXJtaW5hdGVkIG9yIHRydW5jYXRlZDogYnJlYWs=)

1importgem

2#gem.print\_envs()#tolistallavailableenvironments

3env=gem.make("game:GuessTheNumber-v0")

4observation,info=env.reset()

5

6whileTrue:

7#(1)Agentacting:

8action=env.sample\_random\_action()

9#action=agent.act(observation)#realactingbyLLMsampling

10

11#(2)Environmentexecution:

12next\_obs,reward,terminated,truncated,info=env.step(action)

13

14#(3)Agentlearning:

15#agent.learn(observation,action,reward)

16

17observation=next\_obs

18ifterminatedortruncated:break

### 2.2 Tasks and tools

Report issue for preceding element

GEM’s core environment components are tasks and tools. Each combination of a task and an optional set of tools constitutes an environment that tests complex capabilities such as reasoning, multi-step planning, and tool use. These environments can therefore be used to benchmark LLMs and to test and develop new algorithms. GEM currently features seven main categories of tasks:

Report issue for preceding element

Math: Solve math problems with chain-of-thought reasoning.Math with image: Solve geometry math problems with images using chain-of-thought reasoning.Code: Generate code to solve competitive programming problems.Game: Multi-turn text-based games adapted from TextArena (Guertler et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib14 "")).QA: General, potentially knowledge-intensive questions (useful for testing search tool capability).ReasoningGym: A unified interface of ReasoningGym (Stojanovski et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib37 "")) which provides 100+100+ single-turn verifiable tasks.Terminal: Perform complex tasks through a containerized terminal environment.Report issue for preceding element

GEM’s modular design simplifies task integration. Math (with images), code, and QA tasks can be integrated by simply providing a new dataset. Terminal tasks require a new Docker file, instructions, and test cases. New games and other custom tasks can be added by inheriting from GEM’s environment base class and defining their state transition and reward logic.
In addition, tasks can be augmented with any combination of tools. GEM currently supports:

Report issue for preceding element

Python: Parses and executes code blocks, returning the stdout or execution error.Search: Parses a query, executes a search against an external engine, and returns the results.MCP: General tool calling to any external servers that conform to the model context protocol.Report issue for preceding element

The use of tools converts single-turn tasks, like Math or ReasoningGym, into multi-turn tasks in which an agent can learn to call tools and adapt based on their output.

Report issue for preceding element

### 2.3 Asynchronous vectorization and autoreset

Report issue for preceding element

To facilitate efficient agent RL training, we support parallel execution of vectorized environments via asynchronous tool calls to collect episodes in batches. In addition to the latency reduction, the use of vectorized environments with autoreset streamlines the experience collection logic. Users can run a single .reset() at the initialization stage and simply continue with .step() in the following agent-environment loop for continuous data generation. In addition, the user code can use the returned terminated flag to prevent value bootstrapping across episode boundaries, ensuring the correctness of critic learning. An illustration of the autoreset mechanism can be found in [Figure˜2](https://arxiv.org/html/2510.01051v1#S2.F2 "In 2.3 Asynchronous vectorization and autoreset ‣ 2 GEM environments ‣ GEM: A Gym for Agentic LLMs").

Report issue for preceding element

![Refer to caption](https://arxiv.org/html/2510.01051v1/x2.png)Figure 2: Illustration of autoreset in vectorized environments. Autoresetting resets the environment automatically after termination, allowing users to collect batches of episodes by simply running .step() without needing more complicated logic such as keeping track of whether individual episodes have terminated.Report issue for preceding element

### 2.4 Wrappers

Report issue for preceding element

Like in OpenAI-Gym, GEM uses wrappers for easy extensibility. Observation wrappers, for example, control how the episode is converted into an observation. Options include observing just the most recent environment output, a concatenation of all previous environment outputs, a concatenation of all previous environment outputs and actions, or some parsed/summarized version of this.
The Python interpreter or database/web search tools are also formulated as wrappers which can be added on top of any specified task environment.

Report issue for preceding element

## 3 Reinforcement learning with GEM

Report issue for preceding element

In this section, we begin by describing the main RL formulations for LLMs, including their respective flexibilities and limitations ( [Section˜3.1](https://arxiv.org/html/2510.01051v1#S3.SS1 "3.1 Preliminary: LLMs as agents ‣ 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs")). Motivated by this, we then present our baseline algorithm which is applicable to the more flexible RL formulation ( [Section˜3.2](https://arxiv.org/html/2510.01051v1#S3.SS2 "3.2 Baseline algorithms ‣ 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs")).

Report issue for preceding element

![Refer to caption](https://arxiv.org/html/2510.01051v1/x3.png)Figure 3: The illustration of different view of agentic RL. Green nodes denote tokens responsible for loss.Report issue for preceding element

### 3.1 Preliminary: LLMs as agents

Report issue for preceding element

There are three main ways of treating LLM-environment interactions in RL algorithms which each have different limitations and strengths:

Report issue for preceding element

Action = Single token ( [Figure˜3](https://arxiv.org/html/2510.01051v1#S3.F3 "In 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs")(a)): The first approach is to treat each token generated by the LLM as an individual action (Ziegler et al., [2019](https://arxiv.org/html/2510.01051v1#bib.bib50 "")). This, however, means that episodes are typically very long (thousands of tokens), and it also requires specifying the reward for the addition of every token, which is difficult to evaluate. Successful applications of RL in this formulation tend to use sparse outcome reward with discount factor γ=1\\gamma=1(Guo et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib15 "")).

Report issue for preceding element

Action = Response ( [Figure˜3](https://arxiv.org/html/2510.01051v1#S3.F3 "In 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs")(b)): To avoid these complications the second approach is to treat a whole response (a sequence of tokens until an EOS) as a single action222Ignoring token-level PPO clipping which has no effect if the updates are on-policy.(Ahmadian et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib2 ""); Liu et al., [2025a](https://arxiv.org/html/2510.01051v1#bib.bib23 "")). In answering math problems for example—currently the most common testbed for RL for LLMs—each episode contains a question and response. With this view all episodes therefore have length 1 and the RL problem essentially degenerates to contextual bandits (Abe et al., [2003](https://arxiv.org/html/2510.01051v1#bib.bib1 "")). This is convenient as it means sample-based advantage estimation methods such as GRPO (Shao et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib33 "")) can be applied efficiently, and these have been demonstrated to be highly effective. Extending to multi-turn episodes (e.g. for games or tool use), however, results in an issue: Multi-turn interactions have episode lengths >1>1, meaning sample-based advantage estimation methods (e.g., Kazemnejad et al. ( [2025](https://arxiv.org/html/2510.01051v1#bib.bib20 ""))) become infeasible (since they require collecting multiple episode completions from each turn (state) in the episode, leading to exponential complexity).

Report issue for preceding element

Action = Whole interaction ( [Figure˜3](https://arxiv.org/html/2510.01051v1#S3.F3 "In 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs")(c)): One approach to make GRPO applicable to multi-turn interactions is to treat the whole interaction as a single action while masking the loss on tool outputs. This view again degenerates the full RL problem back to one-step RL or contextual bandits, meaning GRPO etc. can be applied. However, it requires two compromises: Firstly, it effectively fixes the discount factor at γ=1\\gamma=1, thus removing the incentive to solve problems quickly. This is significant, for example in [Section˜4.2](https://arxiv.org/html/2510.01051v1#S4.SS2 "4.2 Discount factor 𝛾 matters ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs") where we show how the optimal search algorithm is only recovered when setting γ<1\\gamma<1. Secondly, this approach is limited to single trajectory-level rewards, losing fine-grained per-turn credit assignment.

Report issue for preceding element

Many prior works make these concessions and use GRPO in multi-turn LLM RL (Cao et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib8 ""); Jiang et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib18 ""); Chen et al., [2025a](https://arxiv.org/html/2510.01051v1#bib.bib9 ""); Jin et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib19 ""); Feng et al., [2025a](https://arxiv.org/html/2510.01051v1#bib.bib11 "")). However, to develop an algorithm compatible with the full RL setting, we go back to the second view (action=response) and employ a simple variant of REINFORCE with Return Batch Normalization (ReBN). Unlike GRPO, this algorithm is compatible with per-step dense rewards and arbitrary discount factors (γ≤1\\gamma\\leq 1), thus making it significantly more flexible for optimizing LLMs in complex, multi-turn interactive settings.

Report issue for preceding element

### 3.2 Baseline algorithms

Report issue for preceding element

We start from the foundational on-policy333Orthogonally, we can also utilize proximal updates (Schulman et al., [2017](https://arxiv.org/html/2510.01051v1#bib.bib32 "")) to improve sample efficiency. policy-gradient method REINFORCE (Williams, [1992](https://arxiv.org/html/2510.01051v1#bib.bib45 "")), which optimizes the following objective:

Report issue for preceding element

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝒥REINFORCE​(θ)=1N​∑n=1N∑t=0T(n)−1Gt(n)​log⁡πθ​(at(n)\|st(n)),\\mathcal{J}\_{\\text{REINFORCE}}(\\theta)=\\frac{1}{N}\\sum\_{n=1}^{N}\\sum\_{t=0}^{T^{(n)}-1}G^{(n)}\_{t}\\log\\pi\_{\\theta}(a^{(n)}\_{t}\|s^{(n)}\_{t}),\\vskip-5.69054pt |  | (1) |

where NN is the batch size, \[s0,a0,s1,…,aT−1\]\[s\_{0},a\_{0},s\_{1},...,a\_{T-1}\] is a sequence of states and actions making up a trajectory in which each sts\_{t} and ata\_{t} is itself a sequence of tokens, and Gt=∑k=tT−1γk−t​rkG\_{t}=\\sum\_{k=t}^{T-1}\\gamma^{k-t}r\_{k} is the return.
Though initially designed for single-turn problems (i.e., T(n)=1T^{(n)}=1), GRPO can be extended to multi-turn tasks by sampling a group of MM trajectories per initial state and normalizing the trajectory-level reward for each group444This is not the original GRPO because we fixed the length bias as noted by Liu et al. ( [2025b](https://arxiv.org/html/2510.01051v1#bib.bib25 "")).:

Report issue for preceding element

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝒥GRPO​(θ)=1N​∑n=1N1M​∑m=1MAGRPO(n,m)​∑t=0T(n,m)−1log⁡πθ​(at(n,m)\|st(n,m)),\\mathcal{J}\_{\\text{GRPO}}(\\theta)=\\frac{1}{N}\\sum\_{n=1}^{N}\\frac{1}{M}\\sum\_{m=1}^{M}A\_{\\text{GRPO}}^{(n,m)}\\sum\_{t=0}^{T^{(n,m)}-1}\\log\\pi\_{\\theta}(a^{(n,m)}\_{t}\|s^{(n,m)}\_{t}),\\vskip-8.53581pt |  | (2) |

where AGRPO(n,m)=(∑t=0T−1rt(n,m)−mean​(𝐑))/std​(𝐑)A^{(n,m)}\_{\\text{GRPO}}=(\\sum\_{t=0}^{T-1}r^{(n,m)}\_{t}-\\text{mean}(\\mathbf{R}))/\\text{std}(\\mathbf{R}) with 𝐑={∑t=0T−1rt(n,m)}m∈\[1,…,M\]\\mathbf{R}=\\{\\sum\_{t=0}^{T-1}r^{(n,m)}\_{t}\\}\_{m\\in\[1,\\dots,M\]}. However, this approach has poor credit assignment for multi-turn problems because all turns in the trajectory share the same advantage estimation, and improving it typically requires tree-like sampling which leads to combinatorial explosion. To bypass the expensive sampling from each turn, we can learn a value function to estimate the return GtG\_{t}, known as critic(Sutton and Barto, [2018](https://arxiv.org/html/2510.01051v1#bib.bib38 "")), which in turn guides the policy learning in the actor-critic architecture. We can compute GAE (Schulman et al., [2015](https://arxiv.org/html/2510.01051v1#bib.bib31 "")) for the advantage actor-critic (A2C) objective:

Report issue for preceding element

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝒥A2C​(θ)=1N​∑n=1N∑t=0T(n)−1AGAE,t(n)​log⁡πθ​(at(n)\|st(n)).\\mathcal{J}\_{\\text{A2C}}(\\theta)=\\frac{1}{N}\\sum\_{n=1}^{N}\\sum\_{t=0}^{T^{(n)}-1}A\_{\\text{GAE},t}^{(n)}\\log\\pi\_{\\theta}(a^{(n)}\_{t}\|s^{(n)}\_{t}).\\vskip-8.53581pt |  | (3) |

To retain the benefits of fine-grained and stable advantage estimation without the combinatorial explosion or learning an additional critic, we instead use Return Batch Normalization (ReBN). For ReBN the per-transition returns GiG\_{i} are normalized over the whole batch of transitions:

Report issue for preceding element

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝒥REINFORCE+ReBN​(θ)=1N​∑n=1N∑t=0T(n)−1AReBN,t(n)​log⁡πθ​(at(n)\|st(n)),\\mathcal{J}\_{\\text{REINFORCE+ReBN}}(\\theta)=\\frac{1}{N}\\sum\_{n=1}^{N}\\sum\_{t=0}^{T^{(n)}-1}A\_{\\text{ReBN},t}^{(n)}\\log\\pi\_{\\theta}(a^{(n)}\_{t}\|s^{(n)}\_{t}),\\vskip-8.53581pt |  | (4) |

where AReBN,t(n)=(Gt(n)−mean​(𝐆))/std​(𝐆)A^{(n)}\_{\\text{ReBN},t}=(G^{(n)}\_{t}-\\text{mean}(\\mathbf{G}))/\\text{std}(\\mathbf{G}), with 𝐆={Gt(n)}n∈\[1,…,N\],t∈\[1,…,T(n)−1\]\\mathbf{G}=\\{G^{(n)}\_{t}\\}\_{n\\in\[1,\\dots,N\],t\\in\[1,\\dots,T^{(n)}-1\]}. Each of these algorithms trains the agent by iterating between two main phases: (A) data collection and (B) policy update. We present the RL loop of [Equation˜4](https://arxiv.org/html/2510.01051v1#S3.E4 "In 3.2 Baseline algorithms ‣ 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs") in [Algorithm˜1](https://arxiv.org/html/2510.01051v1#alg1 "In Appendix C Algorithm ‣ GEM: A Gym for Agentic LLMs") in [Appendix˜C](https://arxiv.org/html/2510.01051v1#A3 "Appendix C Algorithm ‣ GEM: A Gym for Agentic LLMs") due to space constraint.

Report issue for preceding element

## 4 Empirical studies with GEM

Report issue for preceding element

In this section, we demonstrate how GEM can facilitate RL research on agentic LLMs through a series of empirical studies. These include a comprehensive apples-to-apples algorithm benchmarking across eight GEM environments ( [Section˜4.1](https://arxiv.org/html/2510.01051v1#S4.SS1 "4.1 Benchmarking RL algorithms for LLMs ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")); analyses of the effects of the discount factor γ\\gamma and tool integration (Sections [4.2](https://arxiv.org/html/2510.01051v1#S4.SS2 "4.2 Discount factor 𝛾 matters ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs") and [4.3](https://arxiv.org/html/2510.01051v1#S4.SS3 "4.3 Tool-integration in math and question-answering tasks ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")); an examination of cross-task generalization ( [Section˜4.4](https://arxiv.org/html/2510.01051v1#S4.SS4 "4.4 Studying generalization ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")); and, finally, a demonstration of GEM’s compatibility with five RL training frameworks along with their easily accessible infrastructure benefits ( [Section˜4.5](https://arxiv.org/html/2510.01051v1#S4.SS5 "4.5 Integration with training frameworks ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")). RL results in a vision-language environment and analysis of a multi-agent environment can be found in [Sections˜D.2](https://arxiv.org/html/2510.01051v1#A4.SS2 "D.2 RL on vision-language environments ‣ Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs") and [D.3](https://arxiv.org/html/2510.01051v1#A4.SS3 "D.3 Multi-agent environments ‣ Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs").

Report issue for preceding element

### 4.1 Benchmarking RL algorithms for LLMs

Report issue for preceding element

Benchmarking has been critical for the progress of RL, with OpenAI-Gym providing standardized environments that enabled systematic evaluation of algorithms (Raffin et al., [2021](https://arxiv.org/html/2510.01051v1#bib.bib29 ""); Huang et al., [2022](https://arxiv.org/html/2510.01051v1#bib.bib17 "")). Following this paradigm, GEM offers a unified testbed for agentic LLMs, where prior work often relied on bespoke tasks that complicate fair comparison. We benchmark all baseline algorithms introduced in [Section˜3.2](https://arxiv.org/html/2510.01051v1#S3.SS2 "3.2 Baseline algorithms ‣ 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs") (GRPO, PPO555PPO in this work generally refers to turn-level PPO instead of token-level PPO commonly seen in single-turn dialogue scenarios (Ouyang et al., [2022](https://arxiv.org/html/2510.01051v1#bib.bib28 ""))., REINFORCE, ReBN) across eight GEM environments under a unified experimental protocol. All algorithms are implemented using Oat (Liu et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib24 "")) with hyperparameters detailed in [Appendix˜F](https://arxiv.org/html/2510.01051v1#A6 "Appendix F Experimental settings ‣ GEM: A Gym for Agentic LLMs"). Results are evaluated by mean episode return, sample efficiency, and stability.

Report issue for preceding element

We present all learning curves in [Figure˜4](https://arxiv.org/html/2510.01051v1#S4.F4 "In 4.1 Benchmarking RL algorithms for LLMs ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs"). We first observe that in all three single-turn environments (labeled with rg), GRPO performs reasonably well, defending its effectiveness in single-step RL with verifiable rewards. However, GRPO falls short when it comes to multi-turn environments (GuessTheNumber and Sudoku), where dense per-turn rewards are available and more fine-grained credit assignment is necessary for efficient policy learning, due to a constant advantage estimation across all steps. Such effects are the most profound when the environment’s reward structure is inherently non-sparse (qa and math is less so).

Report issue for preceding element

In contrast to GRPO, REINFORCE and PPO are natively suitable for multi-turn RL. We find that vanilla REINFORCE is readily a strong baseline in most environments, but it might suffer from suboptimal convergence (e.g., two Sudoku environments). We hypothesize that this might be because the raw return calculation of vanilla REINFORCE can be sensitive to reward shaping, thus hindering exploration; we defer an in-depth ablation study to [Section˜D.1](https://arxiv.org/html/2510.01051v1#A4.SS1 "D.1 Improving learning efficiency via return batch normalization (ReBN) ‣ Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs"). On the other hand, PPO is generally performant, attaining the best episode return in the complex and long-horizon Sudoku environment. This performance advantage can be attributed to a well-learned critic, but it is also deemed difficult to robustly learn an accurate critic (Van Hasselt et al., [2018](https://arxiv.org/html/2510.01051v1#bib.bib42 ""); Kazemnejad et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib20 "")) (as evidenced by the poor performance of PPO in Minesweeper), inviting future works to go in this direction.

Report issue for preceding element

![Refer to caption](https://arxiv.org/html/2510.01051v1/x4.png)Figure 4: Algorithm benchmarking using eight representative environments from GEM. All agents are trained from Qwen3-{scale}-Base models, with scale specified in each plot. rg refers to single-turn reasoning tasks from ReasoningGym; game consists of long-horizon language games; qa and math are tool-integrated multi-turn environments.Report issue for preceding element

Finally, we investigate the proposed REINFORCE variant, which incorporates a simple Return Batch Normalization (ReBN) technique. Results in both [Figures˜1](https://arxiv.org/html/2510.01051v1#S0.F1 "In GEM: A Gym for Agentic LLMs") and [4](https://arxiv.org/html/2510.01051v1#S4.F4 "Figure 4 ‣ 4.1 Benchmarking RL algorithms for LLMs ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs") show that ReBN consistently improves on vanilla REINFORCE by a large margin, suggesting the empirical benefits of adaptive normalization of policy gradient coefficients. Moreover, ReBN outperforms or is comparable with PPO and GRPO in all evaluated environments, rendering it the strongest baseline without expensive computations, such as critic learning or extensive rollouts.

Report issue for preceding element

### 4.2 Discount factor γ\\gamma matters

Report issue for preceding element![Refer to caption](https://arxiv.org/html/2510.01051v1/x5.png)Figure 5: (a) Average number of turns and episode return when trained with different discount factors. (b) Comparative experiment results on tool availability.Report issue for preceding element

Next, we investigate the effect of the discount factor γ\\gamma. A key motivation for REINFORCE+ReBN over GRPO is its compatibility with arbitrary discount factors.
To investigate the effect of this we trained the Qwen3-1.7B-Base model  (Yang et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib48 "")) using REINFORCE+ReBN on the GuessTheNumber environment.
In this environment the agent must guess a hidden number randomly selected between 1 and 50. At each turn the agent may guess, and receives feedback as to whether the hidden number is larger or smaller. The optimal strategy is therefore binary search.

Report issue for preceding element

As shown in [Figure˜5](https://arxiv.org/html/2510.01051v1#S4.F5 "In 4.2 Discount factor 𝛾 matters ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")(a), as expected, smaller γ\\gamma values naturally encourage solutions with fewer turns and drive convergence to the optimal turn count (log2⁡(50)≈5.6\\log\_{2}(50)\\approx 5.6)—achievable only through binary search. Example interactions are included in [Appendix˜B](https://arxiv.org/html/2510.01051v1#A2 "Appendix B Case studies of language games ‣ GEM: A Gym for Agentic LLMs"). As discussed in [Section˜3.2](https://arxiv.org/html/2510.01051v1#S3.SS2 "3.2 Baseline algorithms ‣ 3 Reinforcement learning with GEM ‣ GEM: A Gym for Agentic LLMs"), the natural efficiency incentive from γ<1\\gamma<1 is not compatible with GRPO. Instead, prior works using GRPO hyperparameter tune the environment’s maximum number of turns to get efficient agent behavior (Xue et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib47 "")).

Report issue for preceding element

### 4.3 Tool-integration in math and question-answering tasks

Report issue for preceding element

GEM is designed with modular support for external tools, enabling seamless integration into a range of tasks. To empirically assess the impact of tool use, we focus on two domains: Math and Question-Answering (QA).

Report issue for preceding element

Table 1: Math benchmark scores for four agents, evaluated with and without tool access and RL training. Note: scores should be interpreted relative to other values here due to sensitivity to the grader code (see [Section˜4.3](https://arxiv.org/html/2510.01051v1#S4.SS3 "4.3 Tool-integration in math and question-answering tasks ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")).

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| Qwen3-4B-Base | Base (no tool) | Base (with tool) | Base + RL (no tool) | Base + RL (with tool) |
| AIME24 | 10.0 | 6.7 | 16.7 | 30.0 |
| AMC | 39.8 | 50.6 | 49.4 | 67.5 |
| MATH500 | 61.0 | 62.4 | 67.4 | 71.0 |
| MinervaMath | 36.4 | 30.1 | 40.1 | 40.4 |
| OlympiadBench | 29.5 | 31.0 | 33.5 | 39.9 |
| Average | 35.3 | 36.2 | 41.4 | 49.8 |

Report issue for preceding element

We first investigate the effect of GEM’s Python tool on Math tasks. Starting from the base model Qwen3-4B-Base, we finetune on the math:Orz57K environment, training two variants: one with Python tool integration and one without. The base model and both finetuned models are then evaluated across five distinct math environments. Hyperparameter details are provided in [Appendix˜F](https://arxiv.org/html/2510.01051v1#A6 "Appendix F Experimental settings ‣ GEM: A Gym for Agentic LLMs"), with the training curve shown in [Figure˜5](https://arxiv.org/html/2510.01051v1#S4.F5 "In 4.2 Discount factor 𝛾 matters ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")(b), and Pass@1 accuracy reported in [Table˜1](https://arxiv.org/html/2510.01051v1#S4.T1 "In 4.3 Tool-integration in math and question-answering tasks ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs").

Report issue for preceding element

The math grader used for reward and evaluation is based on HuggingFace’s math\_verify library666 [github.com/huggingface/Math-Verify](https://github.com/huggingface/Math-Verify "").
We found that even minor differences in grading logic across codebases yields substantial variation in reported performance. Thus, all results should be interpreted comparatively—within a consistent evaluation framework—rather than as absolute values. This further highlights the need for unified benchmarking, as provided by GEM.

Report issue for preceding element

Results in [Table˜1](https://arxiv.org/html/2510.01051v1#S4.T1 "In 4.3 Tool-integration in math and question-answering tasks ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs") reveal a clear and consistent pattern: across all environments, performance improves substantially after RL training compared to the base model. Furthermore, the model with access to the Python tool achieves higher final performance in every setting.

Report issue for preceding element

Table 2: QA benchmark scores for the base agent and agents trained with different RL configurations. † and \* denote single-hop and multi-hop datasets, respectively.

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B | |     |
| --- |
| Base |
| (no tool) | | |     |
| --- |
| Base + RL |
| (no tool, |
| single env) | | |     |
| --- |
| Base + RL |
| (no tool, |
| mixed env) | | |     |
| --- |
| Base + RL |
| (with tool, |
| single env) | | |     |
| --- |
| Base + RL |
| (with tool, |
| mixed env) | |
| NQ† | 6.1 | 15.4 | 15.8 | 35.0 | 37.3 |
| TriviaQA† | 35.4 | 43.4 | 44.9 | 69.0 | 71.9 |
| PopQA† | 11.3 | 19.0 | 19.9 | 47.1 | 48.1 |
| HotpotQA\* | 11.1 | 21.1 | 22.1 | 43.2 | 45.5 |
| 2wiki\* | 10.0 | 26.8 | 30.1 | 44.5 | 46.7 |
| Musique\* | 2.9 | 4.7 | 5.5 | 17.6 | 19.9 |
| Bamboogle\* | 17.6 | 28.8 | 28.8 | 49.6 | 48.8 |
| Average | 10.2 | 22.7 | 23.9 | 43.7 | 45.5 |

Report issue for preceding element

We also perform a parallel analysis for QA tasks, this time integrating the Search tool. We train on two environment compositions: qa:HotpotQA alone, and a mixture of both qa:HotpotQA and qa:NaturalQuestions. All other setting are the same as for the Math experiments (see above). Evaluation spans seven diverse QA environments. Results, summarized in [Table˜2](https://arxiv.org/html/2510.01051v1#S4.T2 "In 4.3 Tool-integration in math and question-answering tasks ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs"), mirror those from the math domain: RL finetuning markedly improves performance, and models equipped with the Search tool achieve the highest accuracy in every scenario.

Report issue for preceding element

The consistency of these findings across both domains (mathematics and QA), tools (Python and Search), and multiple evaluation environments underscores the flexibility and robustness of GEM’s approach to RL LLM with tool integration.

Report issue for preceding element

### 4.4 Studying generalization

Report issue for preceding element![Refer to caption](https://arxiv.org/html/2510.01051v1/x6.png)Figure 6: Training on the game:sudoku-v0-easy environment generalizes to ReasoningGym.Report issue for preceding element

GEM’s environments can be used for both training and evaluation. This makes it ideal for investigating cross-environment generalization.
For instance, we demonstrate training on the game:sudoku-v0-easy environment, while periodically evaluating on three different environments, with some encouraging initial generalization results shown in [Figure˜6](https://arxiv.org/html/2510.01051v1#S4.F6 "In 4.4 Studying generalization ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs").

Report issue for preceding element

### 4.5 Integration with training frameworks

Report issue for preceding element

Finally, we demonstrate that GEM—which takes care of the environment side—can be easily integrated with five popular frameworks that handle the training side.
There has been a proliferation of frameworks focusing on the training side of LLM RL. These often rely heavily on multiple other libraries (such as vLLM for response generation (Kwon et al., [2023](https://arxiv.org/html/2510.01051v1#bib.bib21 "")), and DeepSpeed for optimization (Rasley et al., [2020](https://arxiv.org/html/2510.01051v1#bib.bib30 ""))). The diverse range of features and design choices make it challenging for researchers to select and adapt a suitable training framework to their specific needs.

Report issue for preceding element

To address this GEM comes with complete, single-file training scripts showing clean integration into five widely used LLM RL frameworks: Oat (Liu et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib24 "")), Verl (Sheng et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib34 "")), OpenRLHF (Hu et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib16 "")), ROLL (Wang et al., [2025a](https://arxiv.org/html/2510.01051v1#bib.bib43 "")), and RL2 (Tan et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib39 "")). These are validated in [Figure˜7](https://arxiv.org/html/2510.01051v1#S4.F7 "In 4.5 Integration with training frameworks ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")(a) where we show the training curve for each. Despite minor differences due to underlying design choices of the frameworks (e.g., different LLM generation engines) and RL stochasticity, all curves exhibit similar trends, demonstrating that GEM is agnostic to training frameworks and validating their implementation equivalence. Furthermore, supporting a wide range of frameworks allows us to effortlessly access their advanced features. For example, enabling the asynchronous rollout in RL2 gives an immediate 2×2\\times gain in wall-clock efficiency ( [Figure˜7](https://arxiv.org/html/2510.01051v1#S4.F7 "In 4.5 Integration with training frameworks ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")(b)).

Report issue for preceding element

![Refer to caption](https://arxiv.org/html/2510.01051v1/x7.png)Figure 7: (a) Training curves on two environments showing successful integration of GEM into five existing frameworks. (b) Asynchronous rollout improves wall-clock efficiency of training Sudoku-solving agents based on Qwen3-4B-Base.Report issue for preceding element

## 5 Agent evaluation with GEM

Report issue for preceding element

In addition to RL training, GEM can serve as a unified evaluation interface to test LLM agents’ performance. In this section, we present two example use cases where we evaluate agents powered by strong LLMs (GPT-5 (OpenAI, [2025](https://arxiv.org/html/2510.01051v1#bib.bib27 "")), Gemini-2.5-Pro (Gemini Team, [2025](https://arxiv.org/html/2510.01051v1#bib.bib13 "")) and Claude-Sonnet-4 (Anthropic, [2025a](https://arxiv.org/html/2510.01051v1#bib.bib4 ""))) on two complex tasks: database operation via model context protocol (MCP) (Anthropic, [2025b](https://arxiv.org/html/2510.01051v1#bib.bib5 "")) and terminal interaction via docker containers, both of which have been added to GEM following [Appendix˜A](https://arxiv.org/html/2510.01051v1#A1 "Appendix A Environment registration ‣ GEM: A Gym for Agentic LLMs").

Report issue for preceding element

### 5.1 General tool use via model context protocol

Report issue for preceding element![Refer to caption](https://arxiv.org/html/2510.01051v1/x8.png)Figure 8: Benchmark results on MCPMark (Postgres subset) and Terminal-Bench (subset) using GEM as a unified evaluation toolkit.Report issue for preceding element

Modern LLM agents often need to interact with external tools, such as search engines, APIs, and code interpreters. To facilitate this, GEM is designed to be compatible with the MCP, which is an open protocol that provides a standardized way for LLMs to communicate with external tools and data sources.

Report issue for preceding element

The MCP architecture consists of an MCP host (the LLM application), an MCP client, and an MCP server (the external tool). By adopting this protocol, GEM allows for "plug-and-play" tool usage, where any tool that implements the MCP server interface can be used by an agent in a GEM environment without custom integration. This significantly simplifies the process of creating tool-augmented LLM agents and opens up a vast ecosystem of potential tools.

Report issue for preceding element

Using a PostgreSQL MCP tool, we assess the agent’s tool-augmented reasoning capabilities using 20 database tasks taken from MCPMark (Team, [2025a](https://arxiv.org/html/2510.01051v1#bib.bib40 "")). We report the average success rate and the average number of turns required to complete the tasks in the left panel of [Figure˜8](https://arxiv.org/html/2510.01051v1#S5.F8 "In 5.1 General tool use via model context protocol ‣ 5 Agent evaluation with GEM ‣ GEM: A Gym for Agentic LLMs") 777Our evaluation relies on the basic response generation API rather than agent frameworks (e.g., LangChain, OpenAI Agent SDK), which may lead to deviations from the original benchmark results.. GPT-5 attains the best success rate with the fewest interactions, while Gemini-2.5-Pro and Claude-Sonnet-4 have slightly lower and varied performance.

Report issue for preceding element

### 5.2 Terminal environment via Docker container

Report issue for preceding element

To support a wider range of tasks, especially those involving complex software dependencies and interactions with the operating system, GEM includes support for environments running inside docker containers.
The integrated terminal environment provides a sandboxed unix operating system where agents can learn to perform tasks using shell commands. This approach provides a high degree of isolation and reproducibility, ensuring that the environment is consistent across different machines.

Report issue for preceding element

We assess the terminal mastery of LLM agents on 5757 tasks sampled from Terminal-Bench (Team, [2025b](https://arxiv.org/html/2510.01051v1#bib.bib41 "")), without any scaffolding. The right panel of [Figure˜8](https://arxiv.org/html/2510.01051v1#S5.F8 "In 5.1 General tool use via model context protocol ‣ 5 Agent evaluation with GEM ‣ GEM: A Gym for Agentic LLMs") reports the average success rate and the number of turns required to complete the tasks. GPT-5 attains the highest success rate with the most efficient interaction, followed by Claude-Sonnet-4 and Gemini-2.5-Pro. The evaluation leverages the same interaction loop used for RL training, highlighting GEM’s role as a unified framework for both reinforcement learning and standardized evaluation.

Report issue for preceding element

## 6 Conclusions

Report issue for preceding element

GEM aims to accelerate agentic LLM research by providing a decoupled and clean library that is agnostic to training frameworks, a unified agent-environment interface and a suite of standardized environments.
In this paper, we introduced the design choices of GEM, the current suite of task domains and tools, features like vectorized environment execution, a simple yet general multi-turn REINFORCE algorithm implemented in five training frameworks, a comprehensive algorithm benchmarking evaluation, and in-depth analysis on several algorithmic details.
We invite the community to enter the era of experience for LLM agent learning, and join us in both using and continuing to develop the GEM framework.

Report issue for preceding element

## References

Report issue for preceding element

- Abe et al. \[2003\]↑
Naoki Abe, Alan W Biermann, and Philip M Long.

Reinforcement learning with immediate rewards and linear hypotheses.

_Algorithmica_, 37(4):263–293, 2003.

- Ahmadian et al. \[2024\]↑
Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker.

Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms.

_arXiv preprint arXiv:2402.14740_, 2024.

- Andrychowicz et al. \[2021\]↑
Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphaël Marinier, Leonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, et al.

What matters for on-policy deep actor-critic methods? a large-scale study.

In _International conference on learning representations_, 2021.

- Anthropic \[2025a\]↑
Anthropic.

System card: Claude opus 4 & claude sonnet 4.

[https://www-cdn.anthropic.com/07b2a3f9902ee19fe39a36ca638e5ae987bc64dd.pdf](https://www-cdn.anthropic.com/07b2a3f9902ee19fe39a36ca638e5ae987bc64dd.pdf ""), 2025a.

- Anthropic \[2025b\]↑
Anthropic.

Model context protocol.

[https://github.com/modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol ""), 2025b.

- Bai et al. \[2025\]↑
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al.

Qwen2. 5-vl technical report.

_arXiv preprint arXiv:2502.13923_, 2025.

- Brockman et al. \[2016\]↑
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba.

Openai gym, 2016.

- Cao et al. \[2025\]↑
Shiyi Cao, Sumanth Hegde, Dacheng Li, Tyler Griggs, Shu Liu, Eric Tang, Jiayi Pan, Xingyao Wang, Akshay Malik, Graham Neubig, Kourosh Hakhamaneshi, Richard Liaw, Philipp Moritz, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica.

Skyrl-v0: Train real-world long-horizon agents via reinforcement learning, 2025.

- Chen et al. \[2025a\]↑
Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng Chen.

Research: Learning to reason with search for llms via reinforcement learning, 2025a.

- Chen et al. \[2025b\]↑
Wentse Chen, Jiayu Chen, Hao Zhu, and Jeff Schneider.

Context-lite multi-turn reinforcement learning for LLM agents.

In _ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models_, 2025b.

URL [https://openreview.net/forum?id=6CE5PLsZdW](https://openreview.net/forum?id=6CE5PLsZdW "").

- Feng et al. \[2025a\]↑
Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong.

Retool: Reinforcement learning for strategic tool use in llms, 2025a.

- Feng et al. \[2025b\]↑
Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An.

Group-in-group policy optimization for llm agent training.

_arXiv preprint arXiv:2505.10978_, 2025b.

- Gemini Team \[2025\]↑
Google Gemini Team.

Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities.

_arXiv preprint arXiv:2507.06261_, 2025.

- Guertler et al. \[2025\]↑
Leon Guertler, Bobby Cheng, Simon Yu, Bo Liu, Leshem Choshen, and Cheston Tan.

Textarena.

_arXiv preprint arXiv:2504.11442_, 2025.

- Guo et al. \[2025\]↑
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.

Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.

_arXiv preprint arXiv:2501.12948_, 2025.

- Hu et al. \[2024\]↑
Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao.

Openrlhf: An easy-to-use, scalable and high-performance rlhf framework.

_arXiv preprint arXiv:2405.11143_, 2024.

- Huang et al. \[2022\]↑
Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, Jeff Braga, Dipam Chakraborty, Kinal Mehta, and João G.M. Araújo.

Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms.

_Journal of Machine Learning Research_, 23(274):1–18, 2022.

URL [http://jmlr.org/papers/v23/21-1342.html](http://jmlr.org/papers/v23/21-1342.html "").

- Jiang et al. \[2025\]↑
Dongfu Jiang, Zhuofeng Li, Yi Lu, Zhiheng Lvu, Ping Nie, Wenhu Chen, Tianyu Pang, and Chao Du.

Verltool, 2025.

URL [https://github.com/TIGER-AI-Lab/verl-tool](https://github.com/TIGER-AI-Lab/verl-tool "").

- Jin et al. \[2025\]↑
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han.

Search-r1: Training llms to reason and leverage search engines with reinforcement learning.

_arXiv preprint arXiv:2503.09516_, 2025.

- Kazemnejad et al. \[2025\]↑
Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux.

Vineppo: Refining credit assignment in rl training of llms.

In _International conference on machine learning_, 2025.

- Kwon et al. \[2023\]↑
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica.

Efficient memory management for large language model serving with pagedattention.

In _Proceedings of the 29th symposium on operating systems principles_, pages 611–626, 2023.

- Lambert et al. \[2024\]↑
Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al.

Tulu 3: Pushing frontiers in open language model post-training.

_arXiv preprint arXiv:2411.15124_, 2024.

- Liu et al. \[2025a\]↑
Bo Liu, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, et al.

Spiral: Self-play on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning.

_arXiv preprint arXiv:2506.24119_, 2025a.

- Liu et al. \[2024\]↑
Zichen Liu, Changyu Chen, Xinyi Wan, Chao Du, Wee Sun Lee, and Min Lin.

Oat: A research-friendly framework for llm online alignment.

[https://github.com/sail-sg/oat](https://github.com/sail-sg/oat ""), 2024.

- Liu et al. \[2025b\]↑
Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin.

Understanding r1-zero-like training: A critical perspective.

In _Conference on Language Modeling (COLM)_, 2025b.

- OpenAI \[2024\]↑
OpenAI.

Openai o1 system card.

_arXiv preprint arXiv:2412.16720_, 2024.

- OpenAI \[2025\]↑
OpenAI.

Gpt-5 system card.

[https://cdn.openai.com/gpt-5-system-card.pdf](https://cdn.openai.com/gpt-5-system-card.pdf ""), 2025.

- Ouyang et al. \[2022\]↑
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.

Training language models to follow instructions with human feedback.

_Advances in neural information processing systems_, 35:27730–27744, 2022.

- Raffin et al. \[2021\]↑
Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann.

Stable-baselines3: Reliable reinforcement learning implementations.

_Journal of Machine Learning Research_, 22(268):1–8, 2021.

URL [http://jmlr.org/papers/v22/20-1364.html](http://jmlr.org/papers/v22/20-1364.html "").

- Rasley et al. \[2020\]↑
Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He.

Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters.

In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_, pages 3505–3506, 2020.

- Schulman et al. \[2015\]↑
John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel.

High-dimensional continuous control using generalized advantage estimation.

_arXiv preprint arXiv:1506.02438_, 2015.

- Schulman et al. \[2017\]↑
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

_arXiv preprint arXiv:1707.06347_, 2017.

- Shao et al. \[2024\]↑
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al.

Deepseekmath: Pushing the limits of mathematical reasoning in open language models.

_arXiv preprint arXiv:2402.03300_, 2024.

- Sheng et al. \[2024\]↑
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu.

Hybridflow: A flexible and efficient rlhf framework.

_arXiv preprint arXiv: 2409.19256_, 2024.

- Silver and Sutton \[2025\]↑
David Silver and Richard S Sutton.

Welcome to the era of experience.

_Google AI_, 1, 2025.

- Singh et al. \[2023\]↑
Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, et al.

Beyond human data: Scaling self-training for problem-solving with language models.

_arXiv preprint arXiv:2312.06585_, 2023.

- Stojanovski et al. \[2025\]↑
Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, and Andreas Köpf.

Reasoning gym: Reasoning environments for reinforcement learning with verifiable rewards.

_arXiv preprint arXiv:2505.24760_, 2025.

- Sutton and Barto \[2018\]↑
Richard S. Sutton and Andrew G. Barto.

_Reinforcement Learning: An Introduction_.

The MIT Press, second edition, 2018.

- Tan et al. \[2025\]↑
Chenmien Tan, Simon Yu, Lanbo Lin, Ze Zhang, Yuanwu Xu, Chenhao Jiang, Tianyuan Yang, Sicong Xie, and Guannan Zhang.

Rl2: Ray less reinforcement learning.

[https://github.com/ChenmienTan/RL2](https://github.com/ChenmienTan/RL2 ""), 2025.

GitHub repository.

- Team \[2025a\]↑
The MCPMark Team.

Mcpmark: Stress-testing comprehensive mcp use.

[https://github.com/eval-sys/mcpmark](https://github.com/eval-sys/mcpmark ""), 2025a.

- Team \[2025b\]↑
The Terminal-Bench Team.

Terminal-bench: A benchmark for ai agents in terminal environments, Apr 2025b.

URL [https://github.com/laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench "").

- Van Hasselt et al. \[2018\]↑
Hado Van Hasselt, Yotam Doron, Florian Strub, Matteo Hessel, Nicolas Sonnerat, and Joseph Modayil.

Deep reinforcement learning and the deadly triad.

_arXiv preprint arXiv:1812.02648_, 2018.

- Wang et al. \[2025a\]↑
Weixun Wang, Shaopan Xiong, Gengru Chen, Wei Gao, Sheng Guo, Yancheng He, Ju Huang, Jiaheng Liu, Zhendong Li, Xiaoyang Li, et al.

Reinforcement learning optimization for large-scale learning: An efficient and user-friendly scaling library.

_arXiv preprint arXiv:2506.06122_, 2025a.

- Wang et al. \[2025b\]↑
Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li.

Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning, 2025b.

- Williams \[1992\]↑
Ronald J Williams.

Simple statistical gradient-following algorithms for connectionist reinforcement learning.

_Machine learning_, 8:229–256, 1992.

- Xiong et al. \[2025\]↑
Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al.

A minimalist approach to llm reasoning: from rejection sampling to reinforce.

_arXiv preprint arXiv:2504.11343_, 2025.

- Xue et al. \[2025\]↑
Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Zejun Ma, and Bo An.

Simpletir: End-to-end reinforcement learning for multi-turn tool-integrated reasoning.

[https://simpletir.notion.site/report](https://simpletir.notion.site/report ""), 2025.

Notion Blog.

- Yang et al. \[2025\]↑
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al.

Qwen3 technical report.

_arXiv preprint arXiv:2505.09388_, 2025.

- Yao et al. \[2024\]↑
Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan.

τ\\tau-bench: A benchmark for tool-agent-user interaction in real-world domains.

_arXiv preprint arXiv:2406.12045_, 2024.

- Ziegler et al. \[2019\]↑
Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving.

Fine-tuning language models from human preferences.

_arXiv preprint arXiv:1909.08593_, 2019.

## Appendix A Environment registration

Report issue for preceding element

GEM enables rapid development of new RL environments. In this section, we illustrate two scenarios: (i) integrating additional datasets into an existing task and (ii) defining a custom task, followed by the procedure for registering these environments for use.

Report issue for preceding element

The following code snippet shows how to add a new dataset for math environment, where the answer verification logic is predefined by GEM and can be reused.

Report issue for preceding element

[⬇](data:text/plain;base64,aW1wb3J0IGdlbQpmcm9tIGdlbS5lbnZzLnJlZ2lzdHJhdGlvbiBpbXBvcnQgcmVnaXN0ZXIKCnJlZ2lzdGVyKAogICAgIm1hdGg6R1NNOEstRXhhbXBsZSIsCiAgICAiZ2VtLmVudnMubWF0aF9lbnY6TWF0aEVudiIsCiAgICBkYXRhc2V0X25hbWU9ImF4b24tcmwvR1NNLThrIiwgIyBIdWdnaW5nRmFjZSBvciBsb2NhbCBkYXRhc2V0IHBhdGgKICAgIHF1ZXN0aW9uX2tleT0icHJvYmxlbSIsCiAgICBhbnN3ZXJfa2V5PSJhbnN3ZXIiLAopCgplbnYgPSBnZW0ubWFrZSgibWF0aDpHU004Sy1FeGFtcGxlIikgIyByZWFkeSB0byB1c2U=)

1importgem

2fromgem.envs.registrationimportregister

3

4register(

5"math:GSM8K-Example",

6"gem.envs.math\_env:MathEnv",

7dataset\_name="axon-rl/GSM-8k",#HuggingFaceorlocaldatasetpath

8question\_key="problem",

9answer\_key="answer",

10)

11

12env=gem.make("math:GSM8K-Example")#readytouse

Next, we demonstrate how to build a new environment from scratch by defining the initial state distribution (in `.reset()`) and the transition and reward functions (in `.step()`) as follows.

Report issue for preceding element

[⬇](data:text/plain;base64,ZnJvbSBnZW0uY29yZSBpbXBvcnQgRW52CmZyb20gZ2VtLmVudnMucmVnaXN0cmF0aW9uIGltcG9ydCByZWdpc3Rlcgpmcm9tIGdlbS51dGlscy5jb25zdGFudHMgaW1wb3J0IFRFUk1JTkFMX1NUQVRFCmZyb20gZ2VtLnV0aWxzLnBhcnNpbmcgaW1wb3J0IGV4dHJhY3RfbGFzdF9ib3hlZF9hbnN3ZXIKCmNsYXNzIFJldmVyc2VTdHJpbmdFbnYoRW52KToKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBzdHJfbGVuOiBpbnQgPSA1KToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLnN0cl9sZW4gPSBzdHJfbGVuCgogICAgZGVmIF9nZXRfaW5zdHJ1Y3Rpb25zKHNlbGYpOgogICAgICAgIHJldHVybiAoCiAgICAgICAgICAgICJZb3UgYXJlIHRhc2tlZCB0byByZXZlcnNlIGEgZ2l2ZW4gc3RyaW5nLlxuIgogICAgICAgICAgICAiWW91IG1heSBwcm92aWRlIHlvdXIgcmVzcG9uc2UgaW4gYW55IG1hbm5lci4gT25seSB0aGUgY29udGVudCB3cmFwcGVkIGluc2lkZSBcXGJveGVke30gd2lsbCBiZSBjb25zaWRlcmVkIGFzIHlvdXIgZmluYWwgYW5zd2VyLlxuIgogICAgICAgICAgICBmIlBsZWFzZSByZXZlcnNlIHRoZSBzdHJpbmc6IHtzZWxmLmd0X3N0cn0uXG4iCiAgICAgICAgKQoKICAgIGRlZiByZXNldChzZWxmLCBzZWVkPU5vbmUpOgogICAgICAgIHN1cGVyKCkucmVzZXQoc2VlZCkKICAgICAgICBjaGFyYWN0ZXJzID0gc3RyaW5nLmFzY2lpX2xldHRlcnMgKyBzdHJpbmcuZGlnaXRzICAjIEEtWiwgYS16LCAwLTkKICAgICAgICBzZWxmLmd0X3N0ciA9ICIiLmpvaW4ocmFuZG9tLmNob2ljZXMoY2hhcmFjdGVycywgaz1zZWxmLnN0cl9sZW4pKQogICAgICAgIHJldHVybiBzZWxmLl9nZXRfaW5zdHJ1Y3Rpb25zKCksIHt9CgogICAgZGVmIHN0ZXAoc2VsZiwgYWN0aW9uKToKICAgICAgICBjbGVhbl9hY3Rpb24gPSBleHRyYWN0X2xhc3RfYm94ZWRfYW5zd2VyKGFjdGlvbikKICAgICAgICBpZiBjbGVhbl9hY3Rpb24gaXMgTm9uZToKICAgICAgICAgICAgcmV3YXJkID0gMAogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHJld2FyZCA9IGZsb2F0KGNsZWFuX2FjdGlvbls6Oi0xXSA9PSBzZWxmLmd0X3N0cikKICAgICAgICByZXR1cm4gVEVSTUlOQUxfU1RBVEUsIHJld2FyZCwgVHJ1ZSwgVHJ1ZSwge30KCgojIFJlZ2lzdGVyIHlvdXIgZW52aXJvbm1lbnQKcmVnaXN0ZXIoImN1c3RvbTpSZXZlcnNlU3RyaW5nIiwgUmV2ZXJzZVN0cmluZ0VudikKCmVudiA9IGdlbS5tYWtlKCJjdXN0b206UmV2ZXJzZVN0cmluZyIp)

1fromgem.coreimportEnv

2fromgem.envs.registrationimportregister

3fromgem.utils.constantsimportTERMINAL\_STATE

4fromgem.utils.parsingimportextract\_last\_boxed\_answer

5

6classReverseStringEnv(Env):

7def\_\_init\_\_(self,str\_len:int=5):

8super().\_\_init\_\_()

9self.str\_len=str\_len

10

11def\_get\_instructions(self):

12return(

13"Youaretaskedtoreverseagivenstring.\\n"

14"Youmayprovideyourresponseinanymanner.Onlythecontentwrappedinside\\\boxed{}willbeconsideredasyourfinalanswer.\\n"

15f"Pleasereversethestring:{self.gt\_str}.\\n"

16)

17

18defreset(self,seed=None):

19super().reset(seed)

20characters=string.ascii\_letters+string.digits#A-Z,a-z,0-9

21self.gt\_str="".join(random.choices(characters,k=self.str\_len))

22returnself.\_get\_instructions(),{}

23

24defstep(self,action):

25clean\_action=extract\_last\_boxed\_answer(action)

26ifclean\_actionisNone:

27reward=0

28else:

29reward=float(clean\_action\[::-1\]==self.gt\_str)

30returnTERMINAL\_STATE,reward,True,True,{}

31

32

33#Registeryourenvironment

34register("custom:ReverseString",ReverseStringEnv)

35

36env=gem.make("custom:ReverseString")

## Appendix B Case studies of language games

Report issue for preceding element

This section illustrates example interactions of game-playing agents trained on GuessTheNumber under different discount factors γ\\gamma. With a near-undiscounted setting (γ=0.999\\gamma=0.999), the agent fails to complete the game within the trial budget, as it lacks incentive to obtain the terminal correctness reward as earlier as possible (see example below).

Report issue for preceding element

(γ=0.999\\gamma=0.999, the agent exhausts all 1010 trials)State: You are playing Guess The Number.
You have to guess the number between 1 and 50 (inclusive) within 50 turns.
⋯\\cdots
As you play, the history of your guesses will be appended below. Use the information to complete the game before you run out of guesses.Enter your first guess to start the game.At turn 1, you guessed 14, and the target number is higher than 14.At turn 2, you guessed 22, and the target number is higher than 22.…At turn 10, you guessed 39, which has been already guessed before.Enter your next guess.Action: Let’s break down the problem step by step:1\. \*\*Initial Guess (Turn 1):\*\* The target number is higher than 14. This means the target number is between 15 and 50 (inclusive).2\. \*\*Initial Guess (Turn 2):\*\* The target number is higher than 22. This means the target number is between 23 and 50 (inclusive).…From the final two steps, the target number is between 37 and 50 (inclusive). The final answer is \\\backslashboxed{38}.Report issue for preceding element

In contrast, when using a more myopic return estimate (γ=0.9\\gamma=0.9), the agent is encouraged to minimize the number of turns, discovers the binary search strategy, and converges to the optimal policy.

Report issue for preceding element

(γ=0.9\\gamma=0.9, the agent learns to conduct binary search)State: You are playing Guess The Number.
You have to guess the number between 1 and 50 (inclusive) within 50 turns. ⋯\\cdots
As you play, the history of your guesses will be appended below. Use the information to complete the game before you run out of guesses.Enter your first guess to start the game.At turn 1, you guessed 25, and the target number is lower than 25.At turn 2, you guessed 12, and the target number is higher than 12.At turn 3, you guessed 18, and the target number is higher than 18.At turn 4, you guessed 21, and the target number is higher than 21.At turn 5, you guessed 23, and the target number is lower than 23.Enter your next guess.Action: To determine the target number between 1 and 50, we can use a binary search strategy, which is effective for narrowing down a range when given hints like ’higher’ or ’lower.’Step-by-Step Reasoning:1\. \*\*Initial Guess (Turn 1):\*\*
\- The target number is lower than 25.
\- This means the target number is in the range \[1, 24\].2\. \*\*Second Guess (Turn 2):\*\*
\- The target number is higher than 12.
\- This means the target number is in the range \[13, 24\].3\. \*\*Third Guess (Turn 3):\*\*
\- The target number is higher than 18.
\- This means the target number is in the range \[19, 24\].4\. \*\*Fourth Guess (Turn 4):\*\*
\- The target number is higher than 21.
\- This means the target number is in the range \[22, 24\].5\. \*\*Fifth Guess (Turn 5):\*\*
\- The target number is lower than 23.
\- This means the target number is in the range \[22, 22\].Final Guess:The target number is in the range \[22, 22\], which means the target number is 22.
Final Answer:\\\backslashboxed{22}Report issue for preceding element

## Appendix C Algorithm

Report issue for preceding element

For completeness we include the full algorithm of ReBN in [Algorithm˜1](https://arxiv.org/html/2510.01051v1#alg1 "In Appendix C Algorithm ‣ GEM: A Gym for Agentic LLMs").

Report issue for preceding element

Algorithm 1 Multi-turn REINFORCE with Return Batch Normalization (ReBN)

1:Policy πθ\\pi\_{\\theta}, Environment ℳ=(𝒮,𝒜,P,R,γ,ρ)\\mathcal{M}=(\\mathcal{S},\\mathcal{A},P,R,\\gamma,\\rho), Batch size BB

2:while not converged do

3:  Reset batch buffer ℬ←∅\\mathcal{B}\\leftarrow\\emptyset

4:while\|ℬ\|≤B\|\\mathcal{B}\|\\leq Bdo

5:// Multi-turn episode collection

6:   Sample the initial state s0∼ρs\_{0}\\sim\\rho

7:for turn t=0,1,…,T−1t=0,1,\\dots,T-1 until terminate do

8:yt∼πθ(⋅\|st)y\_{t}\\sim\\pi\_{\\theta}(\\cdot\|s\_{t})⊳\\triangleright Generate reasoning + action

9:at←extract\_action​(yt)a\_{t}\\leftarrow\\text{extract\\\_action}(y\_{t})

10:rt←R​(st,at)r\_{t}\\leftarrow R(s\_{t},a\_{t})

11:st+1←P​(st,at)s\_{t+1}\\leftarrow P(s\_{t},a\_{t})

12:endfor

13:fort=0,1,…,T−1t=0,1,\\dots,T-1do

14:Gt←∑k=tT−1γk−t​rkG\_{t}\\leftarrow\\sum\_{k=t}^{T-1}\\gamma^{k-t}r\_{k}⊳\\triangleright Compute discounted return

15:     Add (st,yt,Gt)(s\_{t},y\_{t},G\_{t}) to ℬ\\mathcal{B}

16:endfor

17:endwhile

18:// Return Batch Normalization

19:Gi~←(Gi−mean⁡(𝐆))/std⁡(𝐆)\\tilde{G\_{i}}\\leftarrow\\left(G\_{i}-\\operatorname{mean}(\\mathbf{G})\\right)/\\operatorname{std}(\\mathbf{G})

20:// Policy optimization⊳\\triangleright Or proximal update for data reuse

21:  Update θ\\theta using Monte Carlo policy gradient ∑i=1BGi~​∇θlog⁡πθ​(yi\|si)\\sum\_{i=1}^{B}\\tilde{G\_{i}}\\nabla\_{\\theta}\\log\\pi\_{\\theta}(y\_{i}\|s\_{i})

22:endwhile

Report issue for preceding element

## Appendix D Extended empirical studies with GEM

Report issue for preceding element![Refer to caption](https://arxiv.org/html/2510.01051v1/x9.png)Figure 9: Learning curves of different reward shaping strategies. (a-b) The average success rate of two environments. (c-d) The corresponding average number of turns taken to solve the tasks, equal to the number of tool calls minus one.Report issue for preceding element

### D.1 Improving learning efficiency via return batch normalization (ReBN)

Report issue for preceding element

As briefly discussed in [Section˜4.1](https://arxiv.org/html/2510.01051v1#S4.SS1 "4.1 Benchmarking RL algorithms for LLMs ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs"), while REINFORCE demonstrates strong performance across most environments, its convergence can be suboptimal in certain cases. To investigate this further, we present an in-depth ablation study here.
Following minimalist principles, we began with the vanilla REINFORCE algorithm and a simple reward scheme: r=1r=1 for correct answers and r=0r=0 otherwise. This approach has been shown effective for single-turn RL training \[Singh et al., [2023](https://arxiv.org/html/2510.01051v1#bib.bib36 ""), Xiong et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib46 "")\]. However, as shown in [Figure˜9](https://arxiv.org/html/2510.01051v1#A4.F9 "In Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")(c) (w.o ReBN), it failed to induce tool usage in multi-turn settings, despite significant amount of initial attempts.

Report issue for preceding element

We hypothesize that this failure arises from the absence of negative gradients under 0/1 reward shaping, which are crucial for efficient learning and exploration. To address this, we introduced negative gradients in two ways: (i) assigning fixed negative rewards (r=1r=1 for correct and r=−1r=-1 for incorrect answers, denoted as Neg rew in [Figure˜9](https://arxiv.org/html/2510.01051v1#A4.F9 "In Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")); and (ii) applying Return Batch Normalization with 0/1 rewards, where Monte Carlo returns in REINFORCE are normalized as described in [Algorithm˜1](https://arxiv.org/html/2510.01051v1#alg1 "In Appendix C Algorithm ‣ GEM: A Gym for Agentic LLMs") (denoted as ReBN in [Figure˜9](https://arxiv.org/html/2510.01051v1#A4.F9 "In Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")). While both 0/1 and ±\\pm1 reward schemes theoretically induce the same optimal policy, they might exhibit markedly different learning dynamics in practice.

Report issue for preceding element

Notably, ReBN demonstrates strong and consistent performance across environments—not only in math and QA tasks ( [Figure˜9](https://arxiv.org/html/2510.01051v1#A4.F9 "In Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")) but also in all other settings ( [Figure˜1](https://arxiv.org/html/2510.01051v1#S0.F1 "In GEM: A Gym for Agentic LLMs")). We also observe that models can be sensitive to fixed reward shaping: for example, Neg rew fails to improve tool use in math:Orz57K, yet leads to tool overuse in qa:HotpotQA, both of which are suboptimal behaviors.

Report issue for preceding element

### D.2 RL on vision-language environments

Report issue for preceding element

In addition to text-only environments, we support visual elements as part of the observation for the agent to understand and take actions.
As a demonstrative example, we build a visual-language environment based on Geometry3k dataset888 [https://huggingface.co/datasets/hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k ""). for training reasoning agent to solve geometry math problems with images input. We RL-tune Qwen2.5-VL-3B/7B-Instruct \[Bai et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib6 "")\] using Dr. GRPO \[Liu et al., [2025b](https://arxiv.org/html/2510.01051v1#bib.bib25 "")\], and the learning curves are shown in [Figure˜10](https://arxiv.org/html/2510.01051v1#A4.F10 "In D.2 RL on vision-language environments ‣ Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs"). An example reasoning trace is shown in [Figure˜11](https://arxiv.org/html/2510.01051v1#A4.F11 "In D.2 RL on vision-language environments ‣ Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs").

Report issue for preceding element

![Refer to caption](https://arxiv.org/html/2510.01051v1/x10.png)Figure 10: Learning curves of vision-language agents. We RL-tune Qwen2.5-VL-3B/7B-Instruct using Dr. GRPO on the math:Geometry3K environment and track their training rewards (left) and validation scores (right).
Report issue for preceding element![Refer to caption](https://arxiv.org/html/2510.01051v1/x11.png)Figure 11: An example problem and the response of a trained agent based on Qwen2.5-VL-7B-Instruct.
Report issue for preceding element

### D.3 Multi-agent environments

Report issue for preceding element![Refer to caption](https://arxiv.org/html/2510.01051v1/x12.png)Figure 12: Multi-agent evaluation on TAU-bench retail. Stronger user simulators (rows) consistently improve agent performance (columns) across model strengths.
Report issue for preceding element

GEM supports multi-agent settings where multiple agents interact within the same environment. This capability enables the development of agents that can collaborate, compete, or simulate realistic interactions with other entities.

Report issue for preceding element

Interface design.
GEM provides a MultiAgentEnv base class that extends the standard Gym API to support multiple agents. The step() and reset() functions operate on dictionaries keyed by agent identifiers:

Report issue for preceding element

[⬇](data:text/plain;base64,ZnJvbSBnZW0uZW52cy5tdWx0aWFnZW50IGltcG9ydCBNdWx0aUFnZW50RW52CgplbnYgPSBNeU11bHRpQWdlbnRFbnYoKQpvYnNlcnZhdGlvbnMsIGluZm9zID0gZW52LnJlc2V0KCkgICMgRGljdFthZ2VudF9pZCAtPiBvYnNlcnZhdGlvbl0KCndoaWxlIG5vdCBkb25lOgogICAgYWN0aW9ucyA9IHthZ2VudF9pZDogYWdlbnQuYWN0KG9icykgZm9yIGFnZW50X2lkLCBvYnMgaW4gb2JzZXJ2YXRpb25zLml0ZW1zKCl9CiAgICBvYnNlcnZhdGlvbnMsIHJld2FyZHMsIHRlcm1pbmF0aW9ucywgdHJ1bmNhdGlvbnMsIGluZm9zID0gZW52LnN0ZXAoYWN0aW9ucykKICAgIGRvbmUgPSBhbGwodGVybWluYXRpb25zLnZhbHVlcygpKQ==)

1fromgem.envs.multiagentimportMultiAgentEnv

2

3env=MyMultiAgentEnv()

4observations,infos=env.reset()#Dict\[agent\_id->observation\]

5

6whilenotdone:

7actions={agent\_id:agent.act(obs)foragent\_id,obsinobservations.items()}

8observations,rewards,terminations,truncations,infos=env.step(actions)

9done=all(terminations.values())

To implement a custom environment, users inherit from MultiAgentEnv and implement observe(agent) and \_process\_actions(actions). The framework handles agent lifecycle management and cumulative rewards tracking. Turn coordination is managed via AgentSelector, which supports two modes: sequential (agents act one at a time in round-robin order) and parallel (all agents act simultaneously). The selector determines which agents are active at each step and automatically advances turns, enabling flexible multi-agent interaction patterns without manual bookkeeping.

Report issue for preceding element

TAU-bench retail integration.
We demonstrate this API by integrating the TAU-bench retail benchmark \[Yao et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib49 "")\], which evaluates conversational agents on customer service tasks. We formulate this as a two-agent environment: an assistant agent using tools (order lookup, product search) and a user agent simulating customer behavior via an LLM. The user simulator is initialized with task instructions and generates queries; the assistant must satisfy these requests before episode termination.

Report issue for preceding element

Impact of user model strength.
A key question in multi-agent RL is: how does simulated user agent capability affect trainable assistant agent learning?
We vary both user and assistant models across three levels: weak (Gemini-2.0-Flash-Lite), medium (GPT-4o-mini), and strong (GPT-4o), yielding 9 configurations to study user-assistant model interactions.

Report issue for preceding element

Evaluating across all 115 tasks from the TAU-bench retail test set ( [Figure˜12](https://arxiv.org/html/2510.01051v1#A4.F12 "In D.3 Multi-agent environments ‣ Appendix D Extended empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs")), we find that stronger user agents consistently improve overall success rates across all assistant agent model strengths. Notably, the strongest assistant (GPT-4o) exhibits the largest absolute performance gains (20% from weak to strong user), achieving 61.7% success with a strong user simulator. Interestingly, a strong user paired with a weak assistant (44.3%) outperforms a weak user paired with a strong assistant (41.7%), demonstrating that improving the user agent is crucial for robust conversational task completion. These results motivate us to develop multi-agent RL to co-evolve user and assistant agents to achieve scalable and autonomous learning.

Report issue for preceding element

## Appendix E Related works

Report issue for preceding element

There is a significant body of work on tool-integrated language models—including SkyRL-v0 \[Cao et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib8 "")\], VerlTool \[Jiang et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib18 "")\], ReCall and ReSearch \[Chen et al., [2025a](https://arxiv.org/html/2510.01051v1#bib.bib9 "")\], Search-R1 \[Jin et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib19 "")\], ReTool \[Feng et al., [2025a](https://arxiv.org/html/2510.01051v1#bib.bib11 "")\], and SimpleTIR \[Xue et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib47 "")\]. A common design pattern in these methods is to collect multi-turn agent-environment interactions as single continuous sequences of tokens of agent actions interleaved with environment outputs. Training then simply involves masking the environment outputs from the loss calculation.

Report issue for preceding element

However, this single-sequence approach presents two significant limitations. First, the state observation is rigidly defined as the complete history of actions and outputs. This restricts the ability to use alternative state representations, such as pruning “thinking” tokens or summarizing the history to avoid exceeding context lengths. Second, this formulation inherently limits the reward structure to a single, trajectory-level signal, preventing the use of finer-grained, per-step rewards, and effectively fixing the discount factor at γ=1\\gamma=1. In [Section˜4.2](https://arxiv.org/html/2510.01051v1#S4.SS2 "4.2 Discount factor 𝛾 matters ‣ 4 Empirical studies with GEM ‣ GEM: A Gym for Agentic LLMs") we demonstrate that γ<1\\gamma<1 is crucial for obtaining the optimal fastest search behavior. By contrast, with trajectory-level rewards, the natural speed incentive from γ<1\\gamma<1 is lost, and hence other works, such as SimpleTIR, must tune and enforce a strict turn-limit to get this behavior.

Report issue for preceding element

To address this, our framework, GEM, is designed for maximum flexibility by collecting trajectories as a sequence of individual transitions (i.e., state, action, reward, next state) as in the full, unsimplified RL formulation.
This design choice enables arbitrary state observation constructions (using observation wrappers), and also preserves compatibility with per-turn rewards and arbitrary discount factors γ≤1\\gamma\\leq 1.
The verl-agent framework \[Feng et al., [2025b](https://arxiv.org/html/2510.01051v1#bib.bib12 "")\] also adopts this transition-wise approach, which enables its implementation of GiGPO \[Feng et al., [2025b](https://arxiv.org/html/2510.01051v1#bib.bib12 "")\], an RL method that utilizes turn-level rewards. While GiGPO collapses to trajectory-level GRPO when observations are unique, it is an example of a type of algorithm that is now straightforward to implement with GEM’s infrastructure.

Report issue for preceding element

There are multiple popular frameworks that focus on the agent training side (e.g., Oat \[Liu et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib24 "")\], Verl \[Sheng et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib34 "")\], OpenRLHF \[Hu et al., [2024](https://arxiv.org/html/2510.01051v1#bib.bib16 "")\], ROLL \[Wang et al., [2025a](https://arxiv.org/html/2510.01051v1#bib.bib43 "")\], and RL2 \[Tan et al., [2025](https://arxiv.org/html/2510.01051v1#bib.bib39 "")\]). Currently, many works that build on these, including verl-agent, RAGEN \[Wang et al., [2025b](https://arxiv.org/html/2510.01051v1#bib.bib44 "")\], Verlog \[Chen et al., [2025b](https://arxiv.org/html/2510.01051v1#bib.bib10 "")\], and many of the works above, add environments by directly modifying the source code. This results in tight coupling between training and environments, and makes it difficult to maintain and reuse the environments for future research. As a result, each codebase tends to support only a small, ad-hoc collection of environments, making it hard to compare different methods. Even environments with the same name are often inconsistent between codebases. GEM addresses this by dealing with all the environment infrastructure, including providing a diverse suite of environments, and corresponding baselines. This makes it easy to keep training and environments decoupled, with the aim of freeing researchers from cumbersome environment development and setup, and thus enabling quicker prototyping and evaluation of new ideas.

Report issue for preceding element

## Appendix F Experimental settings

Report issue for preceding element

All our experiments are performed on 8 ×\\times A100 GPUs and finished in about one day. The detailed experimental configurations are shown in [Table˜3](https://arxiv.org/html/2510.01051v1#A6.T3 "In Appendix F Experimental settings ‣ GEM: A Gym for Agentic LLMs").

Report issue for preceding element

Table 3: Hyperparameter configurations used in all experiments.

|     |     |
| --- | --- |
| Parameter | Value |
| Actor |
| Maximum response length per turn | 40964096 tokens |
| Sampling temperature, train | 1.0 |
| Sampling temperature, evaluation | 0.0 |
| (top P, top k) | (1.0, -1) |
| Learner |
| Optimizer | AdamW |
| Adam parameters (β1,β2\\beta\_{1},\\beta\_{2}) | (0.9, 0.95) |
| Weight decay | 0.0 |
| Gradient norm clipping | 1.0 |
| Learning rate scheduler | Constant |
| Learning rate | 1×10−61\\times 10^{-6} |
| Inner proximal update epoch | 2 |
| KL loss coefficient | 0.0 |
| KL penalty coefficient | 0.0 |
| Policy clipping parameter | 0.2 |
| Discount factor | 0.9 (game,qa); 1.0 (otherwise) |
| GAE λ\\lambda | 0.95 |
| Steps | 500 |

Report issue for preceding element

Report Issue

##### Report Github Issue

Title:Content selection saved. Describe the issue below:Description:

Submit without GithubSubmit in Github

Report Issue for Selection

Generated by
[L\\
A\\
T\\
Exml![[LOGO]](<Base64-Image-Removed>)](https://math.nist.gov/~BMiller/LaTeXML/)

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

- Click the "Report Issue" button.
- Open a report feedback form via keyboard, use " **Ctrl + ?**".
- Make a text selection and click the "Report Issue for Selection" button near your cursor.
- You can use Alt+Y to toggle on and Alt+Shift+Y to toggle off accessible reporting links at each section.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).
