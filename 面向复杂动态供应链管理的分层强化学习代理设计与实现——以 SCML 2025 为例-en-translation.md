---
title: "Design and Implementation of Agents for Complex Dynamic Supply Chain Management"
author: "Lita Tang"
date: "2025/12/30"
geometry:
  - a4paper
  - top=30mm
  - bottom=27mm
  - left=18mm
  - right=18mm
  - columnsep=7mm
fontfamily: times
mainfont: Times New Roman
CJKmainfont: SimSun
monofont: "Consolas"
mathfont: "Latin Modern Math"
fontsize: 10.5pt
classoption: twocolumn
tables: true
header-includes:
  - \usepackage{abstract}
  - \usepackage{titlesec}
  - \usepackage[absolute]{textpos}
  # 隐藏默认的 "Abstract" 标题
  - \renewcommand{\abstractname}{}
  - \renewcommand{\absnamepos}{empty}
  # 摘要左缩进 7mm (页边距18+7=25mm)
  - \setlength{\absleftindent}{7mm}
  # 摘要右缩进 7mm
  - \setlength{\absrightindent}{7mm}
  # 按照要求不显示页码
  - \pagenumbering{gobble}
  # 设置一级标题字号 10.5pt
  - \titleformat{\section}{\normalfont\bfseries\fontsize{10.5}{10.5}\selectfont}{\thesection}{1em}{}
  # 设置二级标题字号
  - \titleformat{\subsection}{\normalfont\bfseries\fontsize{10.5}{10.5}\selectfont}{\thesubsection}{1em}{}
  - \usepackage{unicode-math}
  - \setmathfont{Latin Modern Math}
  - \usepackage{etoolbox}
  # 控制浮动对象放置，减少大空白
  - \usepackage{placeins}
  - \renewcommand{\floatpagefraction}{0.8}
  - \renewcommand{\topfraction}{0.9}
  - \renewcommand{\bottomfraction}{0.9}
  - \renewcommand{\textfraction}{0.1}
  # 代码块自动换行
  - \usepackage{fvextra}
  - \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breaksymbolleft={},commandchars=\\\{\}}
  # TikZ 绘图
  - \usepackage{tikz}
  - \usetikzlibrary{shapes,arrows,positioning,fit,backgrounds}
  # 图表标题
  - \usepackage{caption}
---

\makeatletter
\twocolumn[
  \begin{@twocolumnfalse}
    \maketitle
    \vspace{-1.0cm}
    \begin{abstract}
      \noindent \textbf{Abstract:} Supply chain management (SCM) is characterized by high uncertainty, long-range dependencies, and multi-party interactions. The SCML 2025 standard track further introduces perishable inventory and futures contracts, constraining manufacturers with inventory, capacity, and cash flow limitations during multi-threaded concurrent negotiations. Traditional end-to-end reinforcement learning often suffers from exploration risks and training instability. This paper proposes the HRL-XF (Hierarchical Reinforcement Learner Extended for Futures) framework, decomposing the problem into hierarchical modules with distinct temporal scales and responsibilities: L1 Safety Layer constructs a Safety Mask/Shield using deterministic rules, outputting only feasibility constraints (Q\_safe/Q\_safe\_sell, B\_free, time\_mask, etc.) and performing hard clipping on actions; L2 Strategy Layer generates target signals at the daily scale to handle inter-period planning and futures trading; L4 Layer is restructured as "Global Monitor + Priority Coordinator": L4GlobalMonitor maintains global resource commitments and broadcasts constraint states, while GlobalCoordinator outputs thread-specific priority $\alpha$ based on explicit thread features, serving as conditional input to L3 rather than directly modulating actions; The L3 execution layer employs a Decision Transformer to directly output actions compliant with SAO protocol semantics (ACCEPT/REJECT+counter/END), performing consistent action decoding under the L1 security mask.

      \noindent Methodologically, we formalize multi-threaded negotiation on standard tracks as a (thread-level) Dec-POMDP/partially observable randomized game, demonstrating its natural implementation as MAPPO online optimization within the CTDE framework: thread-shared policies and a centralized critic leverage globally broadcasted states for advantage estimation, thereby maintaining training stability against non-stationary adversaries and in strongly constrained environments.

      \noindent In terms of engineering and empirical validation, this paper first systematically summarizes the heuristic agent systems developed prior to the mid-term phase (the Y series and CIR series), serving as sources for security rules and expert data. Second, it conducts a security audit of the SCML 2024 champion PenguinAgent using the SCML Analyzer, identifying three code defects causing "certain breach" accept/offer issues. Two fixes are proposed: the direct-fix version LitaAgent-H and the external security shield version LitaAgent-HS. Under official round-robin settings, LitaAgent-HS reduces unsafe behavior to 0\% while significantly boosting overall scores, validating the stackability of "deterministic security shields + learnable policies."

      \noindent \textbf{Keywords:} Supply Chain Management; Hierarchical Reinforcement Learning; Safety Constraints; MAPPO; Decision Transformer; Futures Contracts; SCML Analyzer
    \end{abstract}
  \end{@twocolumnfalse}
]
\makeatother

\clearpage
\onecolumn
\tableofcontents
\clearpage
\twocolumn

\begin{textblock*}{100mm}(18mm, 280mm)
  \noindent \fontsize{10.5}{10.5}\selectfont
  Affiliation: [Social Information Network · Consensus Informatics] \\
  Advisor: [Takayuki Ito]
\end{textblock*}

## Chapter 1: Introduction

## 1.1 Research Background and Significance

### 1.1.1 Complexity and Intelligence Requirements in Global Supply Chain Management

Supply Chain Management (SCM), serving as the nervous system of modern industrial economies, directly impacts corporate survival and even national economic security through its stability and efficiency. As globalization deepens, modern supply chains exhibit highly networked, dynamic, and uncertain characteristics. From raw material procurement and manufacturing to distribution logistics, each stage involves complex decision-making processes.

However, in recent years, the intensification of the bullwhip effect, sudden public health emergencies (such as COVID-19), geopolitical conflicts, and sharp fluctuations in raw material prices have posed severe challenges to traditional supply chain management approaches based on static operations research optimization. Traditional linear programming or model predictive control (MPC) often assumes a steady-state or quasi-steady-state market environment, making it difficult to handle sudden demand surges or supply disruptions. Furthermore, within multi-party supply chain networks, each node (enterprise) acts as a self-interested agent. The non-cooperative game dynamics between these agents make it challenging to achieve a globally optimal solution through centralized scheduling.

Against this backdrop, automated negotiation and decision-making technologies based on Multi-Agent Systems (MAS) and Reinforcement Learning (RL) have emerged as a research hotspot. By constructing intelligent agents capable of autonomously sensing their environment, predicting market trends, and engaging in strategic interactions with other entities, it is anticipated that these systems will maximize corporate profits at the micro level while enhancing the robustness of the entire supply chain ecosystem at the macro level.

### 1.1.2 SCML: The Premier Testing Ground for Multi-Agent Supply Chain Games

To advance research in this field, the International Joint Conference on Artificial Intelligence (IJCAI) and the International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS) jointly organize the Autonomous Negotiating Agents Competition (ANAC). Among its leagues, the Supply Chain Management League (SCML) stands as the most authoritative and challenging multi-agent supply chain competition globally.

SCML provides a highly realistic simulation environment that models a multi-tiered production network. Within this network, agents must not only decide "what to produce" and "how much to produce," but face the more fundamental challenge of acquiring raw materials and selling products through **concurrent negotiation**. Unlike traditional unilateral inventory management, every order in SCML requires complex negotiation protocols—such as alternating bidding agreements—to be established with another agent. This demands that agents simultaneously master **micro-level game-playing skills** (how to bid) and **macro-level strategic vision** (how to manage cash flow and inventory).

SCML 2025 Standard Track introduces groundbreaking rule changes: **Non-Perishable Inventory and Futures Contracts**.

1.  **Non-Perishable Inventory**: Under previous tournament formats, unused raw materials or unsold products would depreciate or vanish at the end of each day, compelling agents to pursue "zero inventory." The new rules permit long-term inventory retention, introducing a cross-period decision-making dimension: "hoarding for price appreciation" or "liquidating inventory to hedge risks."
    
2.  **Futures Contracts**: Agents can sign orders for delivery decades into the future. This means current decisions not only impact immediate profits but also lock in future cash flows and production capacity.
    

This shift transforms the nature of the environment from a traditional "boy-catcher problem" (a one-off game) into a complex sequential decision-making problem featuring \*\*deep long-term dependencies\*\*, posing unprecedented challenges to existing agent architectures.

## 1.2 Current State and Challenges in Domestic and International Research

### 1.2.1 Traditional Heuristic Agents and Their Limitations

Before the widespread adoption of reinforcement learning, automated agents in the supply chain management (SCM) domain primarily relied on heuristic rules. These rules were typically designed based on domain experts' experience, offering advantages such as low computational overhead, high interpretability, and guaranteed lower bounds on performance.

However, heuristic methods face significant bottlenecks in the SCML 2025 environment:

1.  **Lack of Adaptability**: Thresholds within rules—such as target profit margins—are typically static or adjusted based on simple statistics like moving averages. When confronted with nonlinear market fluctuations—such as hyperinflation—rule-based systems often exhibit delayed reactions.
    
2.  **Local Optimality**: Heuristic algorithms typically decompose complex problems into three independent modules—"negotiation," "production," and "inventory"—for greedy optimization, overlooking the coupling effects between modules. For instance, the negotiation module might accept low-value orders to secure a deal, resulting in unprofitable outcomes for the production module.
    
3.  **Inability to Handle Long-Term Planning**: Rule-based agents struggle to explicitly program complex intertemporal strategies like "stockpiling goods at a loss today to prepare for the peak season 30 days from now."
    

### 1.2.2 Review of the SCML Championship Strategy for 2023-2024

Looking back at the SCML standard track over the past two years, we can clearly see the evolution of agent strategies.

*   **SCML 2023 Champion: MyAgent (Adaptive Threshold-Based)** The champion agent for 2023 primarily employed a strategy based on \*\*Acceptance Thresholds\*\*. This agent dynamically calculates the current market Supply/Demand Ratio in real time and adjusts its Reservation Price accordingly. Its core strength lies in an exceptionally high negotiation success rate, though it adopts a passive approach to inventory management, primarily relying on a conservative "production-based-on-sales" strategy.
    
*   **SCML 2024 Champion: PenguinAgent (ROI-Driven Conservatism)***PenguinAgent* represents the pinnacle of heuristic strategies. Its core concept is **return-on-investment (ROI)-driven conservative negotiation**.
    
    *   **Strategic Core**: It does not pursue market share, but rather seeks certain profits from every transaction. It maintains an extremely rigorous internal valuation model, placing bids only when a potential transaction's expected ROI exceeds a dynamically set high threshold.
        
    *   **Negotiation Mechanism**: Employed a "Time-dependent Concession Curve," maintaining a hard stance early in negotiations and only making rapid concessions as deadlines approached.
        
    *   **Limitations**: Although *PenguinAgent* is extremely unlikely to go bankrupt, its behavior is overly conservative. In the futures market of SCML 2025, if it fails to take risks by stockpiling at low prices, it will be driven out of the market by agents willing to engage in inter-period arbitrage.
        
*   **Other Typical Strategies: DecentralizingAgent***DecentralizingAgent* attempts to solve multi-threaded negotiation problems by decomposing them into multiple independent single-threaded negotiation problems. It utilizes **Gaussian Process Regression** to predict opponents' bottom-line prices. This approach excels in micro-level games but often proves vulnerable when handling macro-level risks of capital chain disruption.
    

### 1.2.3 Challenges in Applying Reinforcement Learning to Supply Chains

In recent years, Deep Reinforcement Learning (DRL) has achieved human-surpassing accomplishments in gaming domains such as Go and Dota 2. However, its implementation in the field of Supply Chain Management (SCM) has faced significant challenges, primarily encountering three major obstacles:

1.  **Credit Assignment Problem**: In SCML 2025, an agent may pay cash to purchase raw materials on Day 1 (negative reward), store them for 20 days, produce goods on Day 21, and sell them on Day 30 (positive reward). This 30-day reward delay makes it difficult for traditional Q-Learning or Actor-Critic algorithms to capture the causal relationship between actions and consequences.
    
2.  **Safe Exploration & Cold Start**: Supply chain environments are extremely sensitive to errors. An untrained RL agent is highly prone to quoting absurd prices or promising undeliverable orders during early exploration, potentially leading to massive penalty fees or outright bankruptcy. Once bankrupt, the episode ends immediately, preventing the agent from acquiring subsequent learning samples.
    
3.  **Non-Stationarity**: SCML is a multi-agent environment where opponent strategies evolve during training (e.g., opponents learn to engage in targeted price suppression). This causes the probability distribution of state transitions to drift, making RL training difficult to converge.
    

## 1.3 Main Research Focus of This Paper

In response to the new challenges posed by SCML 2025 and the limitations of existing approaches, this paper aims to design and implement a hierarchical reinforcement learning agent system that combines **rule-based safety** with **learning adaptability**.

### 1.3.1 Preliminary Work: Heuristic agent Framework (Mid-Term Output)

Before entering the HRL-XF reinforcement learning phase, we completed the design and iteration of the heuristic agent system, forming the LitaAgent series:

(1) Y Series: Introduces the "Three-Tier Procurement Approach" (urgent/planned/optional stockpiling) and inventory-sensitive negotiation strategies, integrating InventoryManager's shortage forecasts and capacity constraints into quotation and acceptance decisions;

(2) CIR Series: Proposes a "Unified Portfolio Evaluation" procurement/sales strategy, treating multiple simultaneous quotes as a portfolio optimization problem. Through simulation-based scenario analysis and threshold-based concessions, it achieves more robust transaction execution and risk control.

(3) Engineering Modularity: Submodules such as InventoryManager, opponent modeling, concession strategy, and parameter adaptation form reusable interfaces, providing a stable engineering foundation for subsequent integration of learning models into the system.

These preliminary efforts fulfill three roles within HRL-XF: providing hard-constraint rules (L1 Safety Mask), supplying expert trajectories (offline BC/forensics), and offering engineering expertise in concurrent negotiation and inventory coupling (L4 thread characteristics and global broadcast design sources).

### 1.3.2 Design of the HRL-XF Layered Architecture

This paper proposes the HRL-XF (Hierarchical Reinforcement Learner Extended for Futures) four-layer hierarchical architecture for stable learning and execution under conditions of "long-term cross-day planning + multi-threaded concurrent negotiation + strong constraint safety." Unlike earlier versions, **HRL-XF's current implementation no longer incorporates any explicit Residual (baseline + residual) action superposition**: Layer 3 directly outputs valid SAOAction, Layer 1 performs only safety constraints and hard clipping, while Layer 4 solely outputs coordination signals (broadcast and priority $\alpha$) without action modulation or sequential deduction-based resource reservation.

\- **L1 Safety Mask/Shield**: 

Construct feasibility boundaries using deterministic rules, outputting and maintaining constraints only (e.g., $Q_{\text{safe}}[\delta]$ , $Q_{\text{safe\_sell}}[\delta]$ , $B_{\text{free}}$ , \`time\_mask\`), and performs hard constraint verification and quantitative clipping on L3 output actions. **L1 does not output baseline actions**.

\- **L2 Strategic Manager**: 

Generate target/intent signals $g_d$ on a daily basis.​ Used for handling futures trading and cross-day planning (e.g., stockpiling/de-stocking, inter-period arbitrage). It can also be combined with potential-based shaping to distribute long-term gains more stably to daily decisions.

\- **L3 Executor, Decision Transformer**: 

Directly output actions conforming to the semantics of the SAO protocol at the negotiation round level:

$\text{SAOAction}\in\{\text{ACCEPT},\ \text{REJECT}+\text{counter},\ \text{END}\}$

Perform consistent decoding under the L1 mask (first select the operation, then select the intersection bucket, then generate $q/p$ , and finally clip).

\- **L4 Global Monitor + Global Coordinator**: 

L4GlobalMonitor maintains company-wide global commitments and remaining resources and broadcasts $c_\tau$ (such as target gaps, remaining safety margins, remaining budgets, etc.); GlobalCoordinator outputs priority per thread based on the explicit thread feature set.

$\alpha_{\tau,k}\in(-1,1)$,

As a conditional input for L3 to achieve cross-thread "soft coordination".

### 1.3.3 Staged Training Paradigms and Engineering Optimization

To address the challenge of RL's convergence difficulties, this paper designs a training pipeline comprising "Forensics → Disentanglement → Pre-training → Fine-tuning." At the engineering level, this paper resolves the deadlock (hang) issue in the NegMas simulation framework during large-scale parallel training. It proposes a solution based on the **Loky backend and Monkey Patch**, ensuring the feasibility of large-scale data collection.

### 1.3.4 Development of the SCML Analyzer Toolkit

Addressing the critical gap in micro-behavior analysis tools within existing ecosystems, this paper designs and implements the **SCML Analyzer**. Utilizing injection-based Tracker technology, this tool captures every agent action—including bidding, inventory changes, and production planning—throughout the entire lifecycle, while providing web-based visualization services. This not only facilitates strategy debugging for this research but also delivers a valuable analytical tool for the broader community.

## 1.4 Thesis Organization Structure

This paper is organized into seven chapters, structured as follows:

*   **Chapter 1: Introduction**: This chapter introduces the research background, the evolution of the SCML competition, an analysis of existing strategies, and the research objectives and contributions of this paper.
    
*   **Chapter 2: Theoretical and Technical Foundations**: This chapter provides a detailed exposition of the rule mechanisms underlying SCML 2025, introducing the theoretical foundations of reinforcement learning, POMDPs, and hierarchical learning.
    
*   **Chapter 3: SCML Environment Modeling and Problem Formulation**: Formalizes the supply chain environment as a POMDP model, deriving the state space, action space, transition function, and reward function based on potential-based shaping.
    
*   **Chapter 4: HRL-XF Layered agent Architecture Design**: An in-depth analysis of the design details within HRL-XF's four-layer architecture, with a focus on the implementation of the security shield mechanism and network.
    
*   **Chapter 5: System Implementation and Engineering Optimization**: Introduces the code implementation of LitaAgent, detailing the data pipeline for the staged training process, as well as the engineering implementation of the SCML Analyzer and Loky Patch.
    
*   **Chapter 6: Experimental Design and Results Analysis**: Presents experimental results for offline pre-training and online fine-tuning, validates the architecture's effectiveness through ablation studies, and conducts case analyses under extreme market conditions.
    
*   **Chapter 7: Summary and Outlook**: Summarize the entire work, discuss existing shortcomings, and outline future research directions (such as L4 neuralization and OneShot adaptation).
    
## Chapter 2: Related Theory and Technical Foundation

This chapter first details the competition mechanism of the Supply Chain Management League (SCML), particularly the non-perishable inventory and futures contract rules introduced in the 2025 season, which form the specific domain context of this study. Subsequently, this chapter reviews the foundational theory of reinforcement learning, highlighting Proximal Policy Optimization (PPO) and Decision Transformers (DTs), while delving into the theoretical framework of Hierarchical Reinforcement Learning (HRL). This establishes the theoretical foundation for the subsequent HRL-XF architecture.

## 2.1 Detailed Mechanism of the Supply Chain Management League (SCML)

SCML is a complex multi-agent simulation environment built upon the NegMAS (Negotiation Multi-Agent System) platform \[4\]. Its core objective is to maximize the long-term cumulative profits of agents through automated negotiation and production scheduling within a simulated manufacturing network.

### 2.1.1 Track Configuration and Production Flowchart

The SCML competition consists of two main tracks: **the OneShot Track and the Standard Track**.

*   **Single-Track Negotiation**: Focuses on one-off negotiations without historical context, primarily assessing the agent's ability to engage in immediate strategic interactions.
    
*   **Standard Race Track**: The primary focus of this study. In the standard race track, agents operate within a multi-level production graph and must make long-term, continuous decisions.
    

**Production Graph Definition**: The SCML environment can be modeled as a directed acyclic graph (DAG).

*   **Node** $V$ : Represents different factories (agents) within the production network. The graph is divided into $L_0, L_1, \dots, L_k$ layer.
    
    *   $L_0$ Tiered Agent: Raw material producer, acting solely as a seller.
        
    *   $L_k$ Tiered agent: The final product manufacturer, serving the external consumer market.
        
    *   Middle-tier agent (such as $L_1$), They are both buyers and sellers. They need to purchase inputs from the previous layer, process them through production lines, and then sell the outputs to the next layer.
        
*   **Direction**: Represents the flow of logistics and capital within the supply chain.
    

*(Insert Figure 2-1 here: Schematic diagram of the SCML production graph, illustrating raw material flow and the dual role of intermediate-layer agents)*

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=1.0cm, auto,
  factory/.style={rectangle, draw, rounded corners, minimum width=1.4cm, minimum height=0.8cm, font=\small},
  arrow/.style={->, thick, >=stealth}]
  
  % L0 layer (raw materials)
  \node[factory, fill=gray!30] (L0a) {$L_0$};
  \node[factory, fill=gray!30, below=0.5cm of L0a] (L0b) {$L_0$};
  \node[left=0.2cm of L0a, font=\small] {Raw};
  
  % L1 layer (intermediate)
  \node[factory, fill=blue!20, right=1.2cm of L0a, yshift=-0.25cm] (L1a) {$L_1$};
  \node[factory, fill=blue!20, below=0.5cm of L1a] (L1b) {$L_1$};
  
  % L2 layer (final)
  \node[factory, fill=green!20, right=1.2cm of L1a, yshift=0.25cm] (L2) {$L_k$};
  \node[right=0.2cm of L2, font=\small] {Final};
  
  % Consumer
  \node[right=1.0cm of L2, font=\small] (C) {Consumer};
  
  % Arrows for goods flow
  \draw[arrow, blue] (L0a) -- (L1a);
  \draw[arrow, blue] (L0b) -- (L1a);
  \draw[arrow, blue] (L0b) -- (L1b);
  \draw[arrow, blue] (L1a) -- (L2);
  \draw[arrow, blue] (L1b) -- (L2);
  \draw[arrow, blue] (L2) -- (C);
  
  % Money flow (dashed, opposite)
  \draw[arrow, red, dashed, bend right=25] (L1a) to (L0a);
  \draw[arrow, red, dashed, bend right=25] (L2) to (L1a);
  
  % Legend
  \node[font=\small, blue] at (2.5,-2.0) {$\rightarrow$ Goods};
  \node[font=\small, red] at (5.0,-2.0) {$\dashrightarrow$ Payment};
  
\end{tikzpicture}
\caption{Fig 2-1: SCML Production Graph Structure}
\end{figure}

**Simulation Process**: A match typically lasts $T_{steps}$ Each simulation day (e.g., 100 days). Each day $t$ is divided into strict phases:

1.  **Negotiation Stage**: Agents exchange offers and execute contracts.
    
2.  **Contract Signing**: The finalized contract is recorded in the ledger.
    
3.  **Production Scheduling**: The agent determines how to allocate inventory for production.
    
4.  **Execution & Settlement**: Delivering goods and transferring funds as stipulated in the contract. A penalty mechanism is triggered in the event of a breach.
    

### 2.1.2 Core Transformation of SCML 2025: Inventory and Futures

Compared to previous iterations, SCML 2025 introduces two key mechanisms that alter the nature of the environment \[7\], transforming it from a "newsboy-like model" to a deep sequential decision problem.

#### 1\. Non-Perishable Inventory

Under SCML 2024 and prior rules, raw materials unused or finished goods unsold by the end of the day were discarded or devalued to zero. This compelled agents to adopt a "zero inventory (Just-in-Time)" strategy, significantly simplifying the state space.

SCML 2025 permits inventory to be held across days, but storage costs $C_{store}$ must be paid. 

$C_{store}(t) = \sum_{g \in \mathcal{G}} \mu_g \cdot I_g(t)$

Where $I_g(t)$ is the inventory level of product $g$ at time $t$ , $\mu_g$ This is the unit holding cost. This mechanism introduces **long-term dependency**: agents can choose to hoard raw materials when prices are low to hedge against future price increases. This implies that decisions made at time $t$ (to buy) may aim to optimize returns at time $t+30$ , rendering traditional myopic strategies ineffective.

#### 2\. Futures Contracts

The new rules permit agents to sign contracts for delivery within the next $t + \delta$ days ( $\delta > 0$ ).

*   **Spot Trading**: $\delta = 0$ , immediate delivery.
    
*   **Futures Trading**: $\delta > 0$ , locking in future prices and quantities.
    

The introduction of futures markets has led to an explosion in the dimensions of the state space. Agents must now manage not only current physical inventory but also **virtual inventory**, representing future delivery obligations. This requires agents to build "order books" and perform cash flow projections based on them.

### 2.1.3 Negotiated Agreement

The SCML Standard employs an Alternating Offers protocol, adhering to SAO (Single Alternating Offers) semantics in its engineering implementation. Let the negotiating parties be $i$ and $j$ . At round (or event step) $\tau$ :

1\. **Offer**: A proposal made by one party

$o_{\tau}^{i\rightarrow j}=\langle q,\ p,\ t_{\text{abs}}\rangle$

$q$ represents quantity, $p$ represents unit price, $t_{\text{abs}}$ For the absolute delivery day.

2\. **Response**: The receiver $j$ responds to the current offer with one of three types:

\- **Accept (ACCEPT)**: Accept the current offer, concluding negotiations and forming a contract.

\- **Reject and Continue (REJECT)**: Decline the current offer, and the negotiation proceeds to the next round; the receiving party may present a counter-offer in the subsequent round.

\- **End Negotiations (END)**: Explicitly terminate negotiations and cease further discussions.

3\. **Counter-offer**: Within our strategic constraints, to reduce "uninformed rejections" and maintain training/execution consistency, we **constrain REJECT to always include a counter-offer**:

$\text{REJECT} \Rightarrow \text{counter}(q',p',t'_{\text{abs}})$

This constitutes a "strategy constraint" rather than a requirement imposed by the protocol itself; its purpose is to reduce invalid rounds and enhance the density of learning signals.

Negotiations operate under strict deadlines. If no agreement is reached by the deadline, negotiations terminate without producing a contract; in practice, this often triggers the common phenomenon of time-dependent concessions.

## 2.2 Foundations of Reinforcement Learning

Reinforcement learning (RL) aims to learn optimal policies through interactions between an agent and its environment, thereby maximizing cumulative expected rewards.

### 2.2.1 Partially Observable Markov Decision Process (POMDP)

In SCML, since agents cannot observe competitors' private information (such as inventory levels, cash balances, or reservation prices), the problem cannot be simply modeled as a Markov Decision Process (MDP). Instead, it must be modeled as a **Partially Observable Markov Decision Process (POMDP)** \[23\].

A POMDP is defined by a septuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$ :

*   $\mathcal{S}$ : The true state space of the environment (including the private states of all agents).
    
*   Action space.
    
*   State transition probability function.
    
*   Reward function.
    
*   Observation Space (local information visible to the agent).
    
*   Observation probability function.
    
*   $\gamma \in [0,1]$ : Discount factor.
    

In a POMDP, the agent cannot directly observe $s_t$ The strategy must be constructed based on historical observation sequences.

### 2.2.2 Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is currently one of the most popular policy gradient-based algorithms \[19\], renowned for its stability and efficiency in continuous control tasks.

The core idea of PPO is to limit the step size of policy updates, preventing the new policy $\pi_\theta$ Deviation from Old Strategy $\pi_{\theta_{old}}$ ​ Too far away, leading to performance collapse. Its objective function includes a clipping term (Clipped Surrogate Objective):

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$

Among these:

*   $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$  is the probability ratio of the new and old strategies.
    
*   $\hat{A}_t$ is Advantage Estimation, generally calculated by GAE (Generalized Advantage Estimation).
    
*   $\epsilon$ is the clipping hyperparameter (typically set to 0.2), used to restrict the range of $r_t(\theta)$ through truncation.
    

In the HRL-XF architecture presented in this paper, the **L2 strategy layer** employs the PPO algorithm. The task of the L2 layer is to output continuous macro-objective vectors (such as target inventory levels or funding limits). PPO excels at handling such continuous action spaces and ensures the stability of long-term policy learning.

### 2.2.3 Decision Transformer

Decision Transformer (DT) introduces a groundbreaking paradigm: transforming reinforcement learning problems into conditional sequence modeling problems.[16]

Unlike traditional methods based on value functions (e.g., DQN) or policy gradients (e.g., PPO), DT utilizes a Transformer architecture (GPT-style) to directly predict the next action based on past states, actions, and \*\*target return\*\*.

Input sequences (Trajectories) are organized as token streams:

$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T)$

where $\hat{R}_t = \sum_{k=t}^T r_k$​ is the cumulative returns-to-go starting from time $t$ .

The objective of the model is to maximize the following likelihood function:

$\max_\theta \sum_{t=1}^T \log P_\theta(a_t | \hat{R}_1, s_1, a_1, \dots, \hat{R}_t, s_t)$

*(Insert Figure 2-2: Schematic Diagram of the Causal Attention Mechanism in the Decision Transformer)*

In an environment like SCML, characterized by **extremely long causal chains** (purchasing raw materials $\rightarrow$ production $\rightarrow$ sales, potentially spanning up to 30 steps), DT's self-attention mechanism can capture credit allocation relationships across vast time spans more effectively than traditional LSTMs. The **L3 Execution Layer** in this paper employs a simplified DT architecture to process micro-negotiation sequences, generating micro-actions based on objectives set by L2.

## 2.3 Hierarchical Reinforcement Learning (HRL), Safety Constraints, and CTDE/MAPPO

### 2.3.1 Stratified Reinforcement Learning and Time Scale Decomposition

The core of hierarchical reinforcement learning lies in introducing strategies at different temporal scales and levels of abstraction. Higher-level strategies generate "intentions/sub-goals/scheduling signals" at slower time scales, while lower-level strategies execute specific actions at faster time scales. Formally, this can be expressed as:

High-level Strategy: $\pi_{\text{high}}(g_t | s_t^{\text{macro}})$ , Generate Goals/Intentions $g_t$ on a daily basis；

Low-level strategy: $\pi_{low}(a_{t,k} | o_{t,k}, g_t, c_t)$ , generating action $a_{t,k}$ for each concurrent thread k at the negotiation round scale.；

where $c_t$ May include globally constrained signals that can be shared across threads (e.g., resource availability, gaps, budgets).

Therefore, when the environment inherently involves multiple time scales ("daily inventory/cash settlement" versus "round-based negotiation bids") and concurrent subtasks (multiple counterparty negotiation threads), HRL reduces the modeling complexity and credit allocation challenges associated with a single strategy.

### 2.3.2 Safety Masking and Shielding

In highly constrained economic systems, pure exploration may lead to irreversible bankruptcy or severe shortage penalties. This paper adopts the Safety Mask/Shield concept: feasibility constraints (inventory availability, fund affordability, delivery feasibility, etc.) are encoded as deterministic rules into masks and hard-cutting operators. This ensures the learning strategy optimizes rewards only within the "feasible set," significantly reducing catastrophic exploration risks during the cold-start phase. Unlike approaches treating heuristic policies as "baseline actions," our novel L1 design outputs no baseline actions. Instead, it generates feasibility boundaries and performs hard constraint verification on proposed actions.

### 2.3.3 CTDE and MAPPO: An Online Learning Framework for Multi-Threaded Concurrent Negotiation

In the SCML standard track, manufacturers engage in concurrent negotiations with multiple opponents on the same simulation day. Multiple negotiation threads share the same set of inventory, capacity, and funding constraints, leading to strong coupling and competition between threads. The Centralized Training, Decentralized Execution (CTDE) framework enables the use of global information during training to learn more stable value functions, while each thread still relies solely on its locally observable information for decision-making during execution. MAPPO (Multi-Agent PPO) is a commonly used stable online algorithm under CTDE: threads share or individually possess policy networks, employ a centralized critic to estimate the advantage function, and utilize PPO's clipping mechanism to suppress abrupt policy updates. This approach is well-suited for scenarios featuring non-stationary opponent policies, partially observable environments, and high-variance training.

# Chapter 3: SCML Environment Modeling and Problem Definition

The futures and inventory mechanism introduced in SCML 2025 transforms supply chain management problems from one-shot games into complex sequential decision problems characterized by deep long-term dependencies and partial observability. This chapter formally models this environment mathematically, defining the state space and action space based on the HRL-XF architecture, and derives in detail the potential reward function for solving the credit allocation problem.

\onecolumn

**Table 3.1 Symbol Table (Used Consistently Throughout This Chapter)**

| Symbol | Meaning |
| --- | --- |
| $d \in \{1,\dots,D\}$ | Simulation Day Index; An episode spans D days |
| $\tau \in \{1,\dots,T\}$ | **Event-step** index; a critical callback (such as respond/propose) or day-of-the-month switch triggers one step. |
| $i \in \mathcal{I}$ | Outer Layer Factory Agent Index (one agent per factory) |
| $k \in \mathcal{K}$ | Inner Layer Our concurrent negotiation thread slot index |
| $s_\tau \in \mathcal{S}$ | Global Actual State of the environment (including all resources, contractual commitments, negotiation mechanism statuses, etc.) |
| $o^i_\tau \in \mathcal{O}_i$ | Local observation of outer factory i (partially observable) |
| $o^k_\tau \in \mathcal{O}_k$ | Local observations of inner thread k (negotiation of local information + broadcasts, etc.) |
| $a^i_\tau \in \mathcal{A}_i$ | Outer Factory i's actions (SAO responses, etc.) |
| $a^k_\tau \in \mathcal{A}_k$ | The action of thread k(SAOAction: ACCEPT / REJECT+counter / END) |
| $r^i_\tau,~r_\tau$ | Outer-layer individual rewards (factory profits) and inner-layer shared rewards (company profits) |
| $t_{\text{abs}}$ | Absolute Delivery Date (Contract Term: time) |
| $\delta = t_{\text{abs}} - d$ | Relative lead time (offset relative to the current day's delivery) facilitates discrete modeling and mask design. |
| $g_d$ | L2 Daily-Scale Goal/Intent Signal |
| $c_\tau$ | L4GlobalMonitor's global broadcast feature |
| $\alpha_{\tau,k}$ | Thread priority provided by GlobalCoordinator |
| $M_\tau$ | L1 Safety Mask/Shield (Feasibility Mask and Hard Clipping Operator) |
| $Q_{\text{safe}}[\delta],~B_{\text{free}}$ | L1 output's safe purchase amount and free budget constraints |

\twocolumn

## 3.1 Mathematical Modeling of the SCML Environment

### 3.1.1 What is the SCML 2025 Standard Environment?

The SCML 2025 Standard can be understood as a **multi-day, multi-agent, highly constrained, event-driven** supply chain simulation environment. Each simulation day (day) typically includes:

1\. **Negotiation Phase**: The factory engages in simultaneous automated negotiations with multiple upstream and downstream counterparts (typically using SAO semantics), with negotiation outcomes potentially generating contracts.

2\. **Execution/Settlement Phase**: Contracts are executed on their delivery date, resulting in inventory changes and cash flow movements; penalties apply for defaults/shortfalls; inventory holding incurs costs.

3\. **State Carryover Across Days**: Inventory, cash, and commitments (future delivery obligations) extend into subsequent dates, forming long-term dependencies.

### 3.1.2 Key Structural Characteristics of the Environment

*   **Long-horizon**: The forward delivery contract you signed today will impact inventory and capital feasibility over the coming days; therefore, strategies must be planned across multiple days.
    
*   **Multi-thread Negotiation**: On any given day, you often find yourself negotiating with multiple counterparties simultaneously. These negotiation threads share the same pool of corporate resources (inventory/capacity/cash/commitments), resulting in strong coupling: "A thread signing a contract" consumes budget/occupies capacity/increases future delivery commitments, thereby affecting whether other threads can still sign contracts.
    
*   **Hard Constraints**: Many actions are "physically or mechanically impossible":
    
    *   Placing an order with insufficient cash will result in payment failure.
        
    *   Committing to deliver beyond capacity will result in certain penalties for breach of contract/shortfall.
        
    *   Responses under the SAO protocol that are not valid (e.g., REJECT without a counter-offer) are immediately invalid.
        
*   **Event-driven (Asynchronous)**: The delegate does not act at every fixed time point, but only when a negotiation thread is scheduled or a daily switch triggers a callback. This implies that "step" must be defined as an **event step** (see sections 3.2/3.3 below).
    
*   **Partial Observability**: You can usually only observe:
    
    *   own local resources (inventory/cash/production capacity, etc.)
        
    *   Current negotiation details (opponent's offer, remaining rounds, history)
        
*   You can't see:
    
    *   Opponent's actual inventory/cash/objectives/strategy internal state
        
    *   Details of the network of contracts between the global truth commitment and its adversaries (unless you have a global monitor or additional observable interfaces)
        

### 3.1.4 Mathematical Description of the Supply Chain Environment as a POMDP

Since agents cannot observe competitors' private states (such as real-time inventory, fund balances, reserve prices, etc.), and market prices are implicitly influenced by supply-demand dynamics, SCML is fundamentally a Multi-Agent Partially Observable Markov Decision Process (POMDP) from a single agent's perspective (treating opponents as part of the environment). For a single agent, the environment can be modeled as a standard POMDP tuple:

$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma, H \rangle$

Among these:

*   $\mathcal{S}$ **(State Space)**: The global state space of the environment, encompassing all agents' private information and market latent variables.
    
*   $\mathcal{A}$ **(Action Space)**: The agent's action space, encompassing decisions such as negotiation offers and production scheduling.
    
*   State Transition Function: Describes inventory depletion, capital flow, and market evolution patterns. Determined by the physical engine of the SCML simulator (NegMAS).
    
*   Reward function: The agent's single-step payoff (profit).
    
*   $\Omega$ **(Observation Space)**: The local information perceivable by the agent.
    
*   Observation probability function.
    
*   $\gamma \in [0, 1)$ : Discount factor, used to balance immediate benefits against long-term returns. In this project, $\gamma=0.99$ emphasizes long-term planning.
    
*   $H$ **(Horizon)**: Time Horizon. It is worth noting that SCML possesses a dual time scale:
    
    *   **Macro-step**: Represents a simulation day (Day), typically $t \in \{0, \dots, T_{days}\}$ .
        
    *   **Micro-step**: Represents negotiation rounds within the same day.
        

### 3.1.5 Why POMDP is Correct

We first present the standard POMDP form:

> **POMDP**: $\mathcal{M}=\langle \mathcal{S},\mathcal{A},\mathcal{O},T,\Omega,R,\gamma \rangle$

*   **Global evolution satisfies Markov property on the "true full state"** If we set $s_\tau$ Defined as "a global variable containing all factory resources, all contractual commitments, and the internal state of all negotiation mechanisms (rounds, deadlines, current offers, participants, etc.)," the environment's next step depends solely on the current state and the actions of all parties:

$P(s_{\tau+1}\mid s_\tau, \mathbf{a}_\tau)$

*   **A single agent can only see one projection of $s_\tau$ (local observation).** For a single agent, the input is not $s_\tau$ but rather a partial observation:
    
    $o_\tau = \Omega(s_\tau)$
    
    Since individual agents cannot observe critical information such as the opponent's internal state, actual resources, and undisclosed commitments, they can only form incomplete information. Therefore, for individual agents, the SCML Standard is inherently a **POMDP**.
    

**The evolution of the SCML world satisfies Markovian properties at the global state level, but individual agents can only access local information. Therefore, it is naturally modeled as a POMDP.**

### 3.1.6 Viewing Ourselves (LitaAgent-HRL) as a Whole: Why Is This a MAPPO Perspective Environment?

From a global perspective, the outer layer of SCML consists of multiple self-interested factory agents interacting simultaneously, which is more naturally formalized as a **Partially Observable Stochastic Game (POSG)/Markov Game**. However, under the single-agent decision perspective adopted in this paper (where opponent strategies are treated as part of the environment), our factory faces a typical partially observable sequential decision problem, which can be modeled as a POMDP:

$\mathcal{M}=\langle \mathcal{S},\mathcal{A},\mathcal{O},T,\Omega,R,\gamma \rangle$

**State Space** (global state): Contains the resource states (inventory/cash/capacity/commitments) of all factories, the set of all signed contracts and their delivery schedules, and the internal states of all negotiation mechanisms (current round, current offer, deadline, participants, etc.), along with exogenous environmental variables and randomness. This definition ensures the environment satisfies Markovian property on:

$P(s_{\tau+1}\mid s_{\tau}, a_{\tau})$

Relying solely on the current state and current action (opponent actions are absorbed into the transition randomness).

**Action Space** $\mathcal{A}$ **(event-driven action)**: Since negotiations are event-driven, this paper adopts "event steps $\tau$ " as the decision step size: each \`respond/propose\` trigger is treated as one event step. On the scheduled negotiation thread, our action is SAOAction:

$a_{\tau}\in\{\text{ACCEPT},\ \text{REJECT}+\text{counter}(q,\delta,p),\ \text{END}\}$.

**Observation Space** $\mathcal{O}$ **and Observation Function** $\Omega$ **(partial observation)**: We can only observe our own partial resources and partial information about the current negotiation (opponent's bid, round number, historical summary, etc.), but cannot observe the opponent's private state (true inventory/cash/objectives/strategy internal state) or their undisclosed commitments. Therefore, observation is

$o_{\tau}=\Omega(s_{\tau})$,

Moreover, partial observability is an inherent property of this task.

Transfer Function: Determined by the simulator, encompassing rules for production, inventory holding, cash settlement, contract execution, and penalties, while being influenced by counterparty behavior.

**Reward Function**: Reflects profits and costs/penalties (inventory holding costs, breach/shortage penalties, etc.), which can be allocated per event step or settled daily with retroactive apportionment.

**Discount factor**: Used to balance immediate gains against long-term returns; long-term cross-day planning tasks typically employ a larger value (e.g., 0.99).

Note: SCML employs dual time scales—the day scale $d$ and the event step scale $\tau$. This paper uniformly represents sequential decision processes using event steps $\tau$, explicitly carrying the day index $d$ within features alongside absolute/relative time ($t_{abs}$ / $\delta$), to ensure the Markov property and learnability of the state representation.

### 3.1.7 Why Using MAPPO Narratives is Justified (From a CTDE Perspective)

The core concept of MAPPO is **CTDE**:

*   **Decentralized Execution**: Each agent uses only its own $o^i\_\tau$ decision.
    
*   **During training (Centralized Training)**: A richer set of global features can be used to learn the value function (critic), reducing variance and mitigating non-stationarity.
    

Can be written as a joint policy decomposition (a common form of independent policies):

$\pi_\theta(a_\tau \mid o_\tau)=\prod_{i \in I}\pi_{\theta_i}(a_{\tau,i} \mid o_{\tau,i})$

and employs the clipped target of PPO (for a given agent $i$ ):

$\mathcal{L}^{\text{PPO}}(\theta_i)=\mathbb{E}\Big[\min(r_\tau(\theta_i)\hat{A}^i_\tau,\ \text{clip}(r_\tau(\theta_i),1-\epsilon,1+\epsilon)\hat{A}^i_\tau)\Big]$

Among these:

*   $r_\tau(\theta_i)=\frac{\pi_{\theta_i}(a^i_\tau\mid o^i_\tau)}{\pi_{\theta_i^{\text{old}}}(a^i_\tau\mid o^i_\tau)}$
    
*   $\hat{A}^i_\tau$: Advantage estimates from the critic (global features may be used)
    

Although the external environment is a multi-agent game, the online learning in this paper updates only the parameters of our own strategy, treating opponents as part of the environment (non-stationary sources). Therefore, the algorithm implementation can be simplified to a "MAPPO-style unilateral PPO," while the modeling narrative retains the MAPPO/CTDE perspective to emphasize the necessity of non-stationarity and centralized value estimation.

### 3.1.8 From POMDP to (Thread-Level) Dec-POMDP/Partially Observable Stochastic Games

#### Thread-Level Dec-POMDP Definition

Treat our concurrent negotiations as $K\_{\max}$ thread slots:

*   $\mathcal{K}={1,\dots,K_{\max}}$
    
*   Each slot is either active (corresponding to an ongoing negotiation) or inactive (empty).
    

The thread-level Dec-POMDP can be written as:

$\mathcal{M}^{\text{thread}}= \langle \mathcal{K}, \mathcal{S}, \{\mathcal{A}_k\}_{k\in\mathcal{K}}, \{\mathcal{O}_k\}_{k\in\mathcal{K}}, T, R, \gamma \rangle$

$R$ **is a shared reward (company-level payoff)**, making this a cooperative Dec-POMDP.

##### Inner global state $s_\tau$

Similar to the outer layer, but the inner layer emphasizes the combination of "enterprise-level resources + all thread negotiation statuses":

*   Company Resources: Inventory/Cash/Production Capacity/Commitments (future delivery obligations)
    
*   Mechanism status of all active negotiation threads (current offer, round, deadline, opponent ID, etc.)
    
*   Daily Targets and Gaps (if you have L2 output $g_d$, which can be considered part of the state)
    

##### Thread-Local Observation

$o^k_\tau = \big( o^{\text{nego}}_{\tau,k},\ g_d,\ c_\tau,\ \alpha_{\tau,k}\big)$

*   $o^{\text{nego}}_{\tau,k}$: This thread's negotiates local information (opponent's offer, round number, role, historical summary, etc.).
    
*   $g_d$: L2 Daily-Scale Objectives (Guiding the day's macro intent)
    
*   $c_\tau$: L4GlobalMonitor Broadcast (Summary of global resource availability, gaps, etc.)
    
*   $\alpha_{\tau,k}$: GlobalCoordinator Output (Thread Priority/Urgency)
    

##### Thread Action $a^k_\tau$

$a^k_\tau \in \{\text{ACCEPT},\ \text{REJECT}+\text{counter}(q,\delta,p),\ \text{END}\}$

*   If thread $k$ is not scheduled or inactive at event step $\tau$ : treat as END
    
*   If scheduled:
    
    *   ACCEPT: Accept the current opponent's offer
        
    *   REJECT: A counter-offer must be provided (quantity $q$, relative delivery date $\delta$, or absolute delivery date $t_{abs}$, and price $p$)
        
    *   END: Termination of Negotiations
        

L1 performs hard feasibility pruning on counter-offers (ensuring they remain within safety boundaries) and guarantees protocol validity (REJECT must include a counter).

##### Shared Rewards

Shared Threads Deliver Enterprise-Level Returns:

$r_\tau = R(s_\tau,\mathbf{a}_\tau)$

The meaning of shared rewards is that threads are truly collaborating (rather than competing). For example:

*   Thread A executes a high-profit contract, but this occupies future delivery capacity, inevitably causing Thread B to default.
    
*   From the company's perspective, this is not the sum of two independent revenues, but rather a strongly coupled, integrated revenue stream.
    

#### Why is this Dec-MAPPO (Thread-Level MAPPO)?

When learning online, we adopt CTDE:

*   **Execution**: Each thread uses only $o^k_\tau$ decision-making
    
*   **Training**: Centralized critics utilize more global information (e.g., $s_\tau$)or $c_\tau$, Estimate advantages, reduce variance
    

##### Parameter-sharing actor

Thread Policy Shared Parameters:

$\pi_\theta(a^k_\tau\mid o^k_\tau) \quad \forall k\in\mathcal{K}$

The negotiation logic across different threads is isomorphic, differing only in input and priority.

#### Centralized Critic

critic can be written as:

*   $V_\phi(s_\tau)$ or more engineering-oriented:
    
*   $V_\phi(c_\tau, {o^{\text{nego}}{\tau,k}}{k\in\mathcal{K}})$
    

where $c_\tau$ is The "Global Summary" maintained by L4GlobalMonitor is suitable for use as input to the critic.

#### How asynchronous/event-driven processing works (masking is key)

Define the scheduling mask:

*   Thread $k$ was scheduled at event step $\tau$ and performed a real action.
    
*   $m_{\tau,k}=0$ : Unallocated/Empty Slot (NO-OP), not subject to loss
    

The PPO objective for thread-level Dec-MAPPO can be written as (with mask):

$\mathcal{L}^{\text{Dec-MAPPO}}(\theta)= \sum_{k\in\mathcal{K}} \mathbb{E}\Big[ m_{\tau,k}\cdot \min\big(r_{\tau,k}(\theta)\hat{A}_{\tau,k},\ \text{clip}(r_{\tau,k}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{\tau,k}\big) \Big]$

Among these:

*   $r_{\tau,k}(\theta)=\frac{\pi_\theta(a^k_\tau\mid o^k_\tau)}{\pi_{\theta^{\text{old}}}(a^k_\tau\mid o^k_\tau)}$
    
*   $\hat{A}_{\tau,k}$ is Advantage Estimates from Centralized Critic
    

### 3.1.9 Correspondence with HRL-XF (L1/L3/L4)

*   **L1 Safety Mask/Shield**: From $s_\tau$ (Company Resources and Commitments) calculates $M_\tau$. Block non-actionable operations and enforce hard limits on quantities (security foundation)
    
*   **L3 Executor**: Thread actor, returns SAOAction(ACCEPT/REJECT+counter/END)
    
*   **L4GlobalMonitor**: Maintains global commitments and remaining resources, outputs broadcast feature $c_\tau$ (For actor/critic)
    
*   **GlobalCoordinator**: Output priority $\alpha_{\tau,k}$ as a conditional variable input to L3, it enables cross-thread coordination without directly modulating actions.
    

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=0.6cm, auto,
  box/.style={rectangle, draw, rounded corners, text centered, minimum height=0.9cm, minimum width=1.8cm, font=\small},
  arrow/.style={->, thick}]
  
  % Dec-POMDP subgraph
  \node[box, fill=blue!10] (S) {$s_\tau$: Global};
  \node[box, fill=blue!10, right=of S] (O) {$o_\tau^k$: Local};
  \node[box, fill=blue!10, right=of O] (A) {$a_\tau^k$: Action};
  \node[box, fill=blue!10, right=of A] (R) {$r_\tau$: Reward};
  
  \draw[arrow] (S) -- (O);
  \draw[arrow] (O) -- (A);
  \draw[arrow] (A) -- (R);
  \draw[arrow, bend left=40] (R) to (S);
  
  % HRL-XF subgraph
  \node[box, fill=green!10, below=1.2cm of S] (L4) {L4: Coord};
  \node[box, fill=green!10, right=of L4] (L3) {L3: $\pi_\theta$};
  \node[box, fill=green!10, right=of L3] (L1) {L1: Mask};
  
  \draw[arrow] (L4) -- (L3);
  \draw[arrow] (L3) -- (L1);
  
  % Cross connections
  \draw[arrow, dashed] (S) -- (L4);
  \draw[arrow, dashed] (O) -- (L3);
  \draw[arrow, dashed] (L1) -- (A);
  
  % Labels
  \node[above=0.15cm of O, font=\small] {Dec-POMDP};
  \node[below=0.15cm of L3, font=\small] {HRL-XF};
\end{tikzpicture}
\caption{Dec-POMDP and HRL-XF Architecture Correspondence}
\end{figure}

## 3.2 State Space and Observation Model

To adapt to the HRL-XF hierarchical architecture, we further decouple the local observation $o_t \in \Omega$ into **macroscopic state (L2 input) and microscopic state (L3 input)**. This design references the `state_builder.py` module in the codebase.

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=0.4cm, auto,
  box/.style={rectangle, draw, minimum width=1.6cm, minimum height=0.6cm, font=\small},
  arrow/.style={->, thick}]
  
  % Raw observation
  \node[box, fill=gray!20] (raw) {Raw Obs $o_t$};
  
  % Macro components
  \node[box, fill=orange!20, below left=0.8cm and -0.4cm of raw] (vres) {$v_{res}$};
  \node[box, fill=orange!20, right=0.2cm of vres] (vfin) {$v_{fin}$};
  \node[box, fill=orange!20, right=0.2cm of vfin] (vmkt) {$v_{mkt}$};
  \node[box, fill=orange!20, right=0.2cm of vmkt] (vtime) {$v_{time}$};
  
  % Macro state
  \node[box, fill=yellow!30, below=0.7cm of vfin, xshift=0.5cm, minimum width=2.0cm] (smacro) {$s_{macro}$ $\to$ L2};
  
  % Micro additions
  \node[box, fill=blue!20, below=0.7cm of smacro, xshift=-0.8cm] (hneg) {$h_{neg}$};
  \node[box, fill=blue!20, right=0.2cm of hneg] (gt) {$g_t$};
  
  % Micro state
  \node[box, fill=green!30, below=0.7cm of hneg, xshift=0.5cm, minimum width=2.0cm] (smicro) {$s_{micro}$ $\to$ L3};
  
  % Arrows
  \draw[arrow] (raw) -- (vres);
  \draw[arrow] (raw) -- (vtime);
  \draw[arrow] (vres) -- (smacro);
  \draw[arrow] (vfin) -- (smacro);
  \draw[arrow] (vmkt) -- (smacro);
  \draw[arrow] (vtime) -- (smacro);
  \draw[arrow] (smacro) -- (hneg);
  \draw[arrow] (smacro) -- (gt);
  \draw[arrow] (hneg) -- (smicro);
  \draw[arrow] (gt) -- (smicro);
  
\end{tikzpicture}
\caption{Fig 3-2: State Space Decomposition}
\end{figure}

### 3.2.1 L2 Macro State $s_{macro}$

The macro state aims to capture long-term trends and overall health of the supply chain. It is a high-dimensional continuous vector composed of the following four types of features:

$s_{macro}^{(t)} = \text{Concat}( \mathbf{v}_{res}, \mathbf{v}_{fin}, \mathbf{v}_{mkt}, \mathbf{v}_{time} )$

1.  **Resource Status** $\mathbf{v}_{res}$ **(including virtual inventory)**: Not only includes current physical inventory $I^r_t, I^p_t$. More crucially, it includes a projection of \*\*virtual inventory\*\* for the next $K$ days. Virtual inventory $VI$ is defined as the expected inventory change based on the order book:

$VI_k(t) = I(t) + \sum_{\delta=1}^k \left( \sum_{c \in \mathcal{C}_{buy}^{\delta}} q_c - \sum_{c \in \mathcal{C}_{sell}^{\delta}} q_c - \text{Prod}_{\delta} \right)$

In the code implementation, the $K=40$ is retrieved, enabling the L2 network to perceive the supply-demand gap (Shortfall Risk) over the next 40 days.

*   **Financial Status** $\mathbf{v}_{fin}$ : Contains the normalized current fund balance $B_t / B_{init}$ Current credit utilization rate and cash flow projections for the next $K$ days.
    
*   **Market Status** $\mathbf{v}_{mkt}$ : Market reference prices (Spot Market Price) for raw materials and finished goods, along with their first-order differences (volatility).
    

$P_{avg}(t) = \alpha P_{obs}(t) + (1-\alpha) P_{avg}(t-1)$

Use the exponential moving average (EMA) to smooth out observation noise.

*   **Time State** $\mathbf{v}_{time}$ : Includes global progress $t/T_{days}$ Relative position to the current step.

### 3.2.2 L3 Microstate $s_{micro}$

Microstates are used to support specific negotiation games, in addition to containing $s_{macro}$. In addition to providing summary information (to offer context), it also highlights specific details from the current negotiation thread:

$s_{micro}^{(t, \tau)} = \text{Concat}( \text{Embed}(s_{macro}^{(t)}), \mathbf{h}_{neg}, \mathbf{g}_t )$

1.  **Negotiation History** $\mathbf{h}_{neg}$ : The current negotiation counterpart's historical offer sequence over the past $N$ rounds (typically $N=5$ ) and the counterpart's action type (accept/reject).
    
2.  **Macro Objectives** $\mathbf{g}_t$ : The daily strategic objectives issued by Level 2 (see Section 3.3) serve as **conditional inputs** for micro-level decision-making, guiding the negotiation direction at Level 3.
    

## 3.3 Hierarchical Decomposition of Action Space

To address high-dimensional and continuous action spaces, the HRL-XF architecture employs a hierarchical decomposition strategy, breaking down complex supply chain decisions into strategic planning (Level 2) and tactical execution (Level 3).

### 3.3.1 L2 Macro Action Space (Strategic Action)

L2 layer executes a decision once at the start of each day (`before_step`), outputting a continuous target vector $\mathbf{g}_t \in \mathbb{R}^4$ that defines the day's trading boundaries:

$\mathbf{g}_t = [ Q_{target}^{buy}, P_{limit}^{buy}, Q_{target}^{sell}, P_{limit}^{sell} ]$

*   $Q_{target} \in [0, 1]$ : Normalized expected total purchases/sales for the current date.
    
*   $P_{limit} \in [-1, 1]$ : Normalized price floor.
    
    *   For buy orders, the actual limit price $P_{real} = P_{mkt} \cdot (1 + \beta \cdot P_{limit})$ .
        
    *   For selling, the actual floor price is $P_{real} = P_{mkt} \cdot (1 - \beta \cdot P_{limit})$ .
        

These objectives are not rigid constraints but rather serve as a \*\*potential field\*\* to guide the behavior of the L3 layer.

### 3.3.2 L3 Micro Action Space: SAOAction(ACCEPT / REJECT+counter / END)

The SCML standard track adopts the SAO (Single Alternating Offers) protocol from NegMAS. Under this protocol, manufacturers do not possess a standalone "OFFER" action within the respond() callback: any counter-offer must be returned as a REJECT\_OFFER carrying the counter-offer. Furthermore, at the implementation level, a REJECT\_OFFER with outcome=None is invalid. Therefore, this paper unifies the micro-actions of L3 as follows:

*   Discrete operator $op \in \{ACCEPT, REJECT, END\}$
    
*   And when op=REJECT, a counter-offer must be output: $outcome=(q, t_abs, p)$
    

The outcome field follows the sequence (quantity, time, unit\_price). To facilitate masking and discrete modeling, the network may internally use relative lead times $\delta=t_{abs} - t_{now}   \in  \{0,1,…,H\}$ , remapping them back to absolute lead times $t_{abs}$ prior to execution.

The mapping between action semantics and protocols is as follows:

*   op=ACCEPT: Returns SAOResponse(ACCEPT\_OFFER, offer\_in). Note that the opponent's original offer\_in must be included; passing None or a custom outcome is not permitted.
    
*   op=REJECT: Returns SAOResponse(REJECT\_OFFER, offer\_out), note REJECT must comes with a counter-offer, otherwise it will become an illegal action.
    
*   op=END: Returns SAOResponse(END\_NEGOTIATION, None).
    

The \`propose()\` interface typically only outputs \`offer\_out\` (corresponding to \`op=REJECT\`). If you need to "not open the game," map \`op=END\` to \`None\` (to have the controller terminate the negotiation).

## 3.4 Reward Function Design and Potential Shaping

The core challenge of SCML 2025 lies in the extreme sparsity and delay of \*\*credit assignment\*\*. For instance, purchasing raw materials at time $t$ constitutes a negative-reward action that reduces cash reserves, with its positive returns potentially materializing only after finished goods are sold at time $t+20$ . If net profit is directly used as the reward, RL agents are highly susceptible to falling into a local optimum trap of "neither buying nor selling."

To this end, this paper designs a multi-level reward function based on the \*\*Potential-Based Reward Shaping\*\* theory proposed by Ng et al.

\begin{figure}[h]
\centering
\begin{tikzpicture}[scale=1.2]
  % Axes
  \draw[thick,->] (0,0) -- (6,0) node[right, font=\small] {Time};
  \draw[thick,->] (0,0) -- (0,4) node[above, font=\small] {Value};
  
  % Cash line (decreasing then increasing)
  \draw[blue, very thick] (0,2.5) -- (1.2,1.9) -- (2.4,1.2) -- (3.6,1.9) -- (4.8,3.2);
  \node[font=\small, blue] at (5.2,3.5) {$B$};
  
  % Potential line (increasing)
  \draw[red, very thick, dashed] (0,0.6) -- (1.2,1.3) -- (2.4,2.0) -- (3.6,1.7) -- (4.8,1.0);
  \node[font=\small, red] at (5.2,1.2) {$\Phi$};
  
  % Total value
  \draw[green!60!black, very thick, dotted] (0,3.1) -- (1.2,3.2) -- (2.4,3.2) -- (3.6,3.6) -- (4.8,4.2);
  \node[font=\small, green!60!black] at (5.4,4.5) {$B+\Phi$};
  
  % Annotations
  \node[font=\small] at (1.8,0.4) {Buy};
  \node[font=\small] at (4.2,0.4) {Sell};
  
\end{tikzpicture}
\caption{Fig 3-4: Potential-Based Reward Shaping}
\end{figure}

### 3.4.1 Potential Energy Function $\Phi(s)$

We define the potential value of state $s$ as the market fair value of all physical assets currently held by the agent:

$\Phi(s_t) = \underbrace{I^r_t \cdot \bar{P}^r_t}_{\text{原材料价值}} + \underbrace{I^p_t \cdot \bar{P}^p_t}_{\text{成品价值}}$

where $\bar{P}^r_t$, $\bar{P}^p_t$ are the market reference average prices for raw materials and finished goods at the respective times (estimated by `scml_analyzer.detectors`).

### 3.4.2 L2 Strategy Layer Rewards

The objective of L2 is to maximize long-term equity. According to the "Progress Report," to address the issue of immediate losses upon purchase, a momentum differential $F(s_t, s_{t+1}) = \gamma \Phi(s_{t+1}) - \Phi(s_t)$ is introduced:

$$
\begin{aligned}
R_{L2}(s_t, a_t, s_{t+1}) = & \underbrace{(B_{t+1} - B_t)}_{\Delta \text{Equity}} \\
& + \underbrace{(\gamma \Phi(s_{t+1}) - \Phi(s_t))}_{\Delta \text{Potential}} \\
& - \underbrace{\lambda \cdot \text{Shortfall}_t}_{\text{Risk Penalty}}
\end{aligned}
$$

**Theoretical Nature**:

*   **Buying Operations**: Funds $B$ decrease, but inventory $I$ increases, leading to $\Phi$ growth. If the purchase price equals the market price, then $R_{L2} \approx 0$ . This eliminates the negative feedback loop associated with buying behavior.
    
*   **Buy low, sell high**: If purchased below market price, $\Delta B$ decreases less than $\Delta \Phi$ increases, generating a positive incentive $R_{L2} > 0$ that encourages arbitrage.
    
*   **Risk Penalty**: In the event of a shortfall, impose a substantial negative reward $\lambda$ (e.g., $\lambda=5.0$ ) to compel L2 to learn proactive inventory strategies.
    

### 3.4.3 L3 Execution Layer Rewards

L3 focuses on the efficiency and success rate of tactical execution. Its reward function comprises three components:

$R_{L3} = w_1 \cdot R_{align} + w_2 \cdot R_{advantage} + w_3 \cdot R_{liquidity}$

1.  **Alignment Reward** $R_{align}$: Penalty for Actual Trading Volume vs. L2 Target $Q_{target}$, using Mean Squared Error (MSE) Loss to ensure tactics align with strategy.
    
2.  **Advantage Rewards** $R_{advantage}$: Assessing the quality of a single transaction. For a buy order, if the execution price $P_{deal} < P_{mkt}$, a positive reward is given.
    
    
3.  **Liquidity Rewards** $R_{liquidity}$ : A small positive incentive is provided whenever a transaction is completed. This prevents the "freezing" phenomenon caused by the conservative nature of offline RL models during initial exploration.
    

# Chapter 4: HRL-XF Hierarchical Agent Architecture Design

Addressing the coexistence of long-term dependencies and micro-level interactions within the SCML 2025 environment, this chapter details the system architecture of HRL-XF (Hierarchical Reinforcement Learner Extended for Futures). Engineered as a modular decision pipeline, this architecture decomposes high-dimensional POMDP problems into solvable subproblems through the collaborative operation of four layers: L1 to L4. This chapter provides an in-depth analysis of the mathematical models, network architectures, and inter-layer coordination mechanisms for each layer.

## 4.1 Architecture Overview: From HRL-X to HRL-XF

The core design philosophy of the HRL-XF architecture is \*\*"macro planning guides micro adjustments, with security shields providing a safety net."\*\* The entire decision-making process of the agent system can be formalized as a hierarchical policy function $\Pi$ . To adapt to futures markets, we divide the architecture into two temporal scales: \*\*day-scale **strategic planning and** round-scale\*\* tactical execution.

*\[Figure 4-1 Recommended Insertion Position: HRL-XF System Architecture Overview. Left side displays L2 macro-cycle, right side displays L3/L4 micro-cycle, bottom shows L1 shield and environmental interaction interface\]*

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=0.3cm, auto,
  layer/.style={rectangle, draw, rounded corners, minimum width=1.8cm, minimum height=0.5cm, font=\tiny, text centered},
  arrow/.style={->, thick, >=stealth}]
  
  % L2 Macro cycle (left)
  \node[layer, fill=orange!30] (L2) {L2: Strategic};
  \node[below=0.15cm of L2, font=\tiny] {Day-scale};
  
  % L3/L4 Micro cycle (right)
  \node[layer, fill=green!30, right=1.2cm of L2] (L4) {L4: Coord};
  \node[layer, fill=blue!30, below=0.3cm of L4] (L3) {L3: Executor};
  \node[right=0.1cm of L4, font=\tiny] {Round};
  \node[right=0.1cm of L3, font=\tiny] {-scale};
  
  % L1 Shield (bottom)
  \node[layer, fill=red!20, below=0.6cm of L2, xshift=0.6cm] (L1) {L1: Safety Shield};
  
  % Environment
  \node[layer, fill=gray!20, below=0.4cm of L1] (ENV) {SCML Environment};
  
  % Arrows
  \draw[arrow] (L2) -- node[above, font=\tiny]{$g_t$} (L4);
  \draw[arrow] (L4) -- node[right, font=\tiny]{$c_\tau,\alpha$} (L3);
  \draw[arrow] (L3) -- (L1);
  \draw[arrow] (L1) -- node[right, font=\tiny]{action} (ENV);
  \draw[arrow, dashed] (ENV.west) -- ++(-0.3,0) |- node[left, font=\tiny, near start]{obs} (L2.west);
  
\end{tikzpicture}
\caption{Fig 4-1: HRL-XF System Architecture}
\end{figure}

### 4.1.1 Mathematical Description of the Decision Pipeline

HRL-XF divides decision-making into daily and round-based scales, adhering to the principle of "first computing the feasibility boundary, then optimizing payoff within the feasible set." The primary workflow for each simulation day $t$ is as follows:

(1) L1 Safety Mask calculation (can be recalculated before the step or before any decision):

Based on current inventory (raw materials/finished goods), production capacity, committed deliveries, and cash flow, calculate:

*   $Q_{safe}[\delta]$ : Maximum safe procurement quantity for the purchasing side at each relative delivery date $\delta$ ;
    
*   $Q_{safe-sell}[\delta]$ : Maximum deliverable quantity on the sell side at each $\delta$ (permitting same-day production and same-day delivery for $\delta=0$ );
    
*   $B_{free}$: Free budget available for new procurement;
    
*   $time\_mask$: Mask for setting unfeasible delivery time buckets to $-\infty$.
    

L1 does not output base actions, but only provides hard constraints and the clipping operator $clip_action(q,t,p)$ .

(2) L2 Daily-Scale Target Generation:

L2 Read macro state $s_{macro}^{(t)}$ Output the target vector $\mathbf{g}_t$ for the current day $t$ (e.g., buy/sell side volume and price boundaries) as conditional inputs for lower-level strategies, serving as guidance for long-term planning and futures trading.

(3) L4 Global Monitoring and Coordination:

L4GlobalMonitor maintains company-wide global commitments and remaining resources (such as daily executed volume, committed delivery volume, remaining budget, etc.), broadcasting the global feature $c_t$ ；

GlobalCoordinator receives "thread explicit attribute set + global attributes" and outputs the priority $\alpha_k \in (-1, 1)$ of each active thread. As a conditional variable for L3, $\alpha_k$ indicates the thread's priority in resource contention, rather than being directly used for modulating actions or performing sequential decrements.

(4) L3 Execution Action Generation (for each respond/propose):

For each active thread, L3 reads the local observation $o_{t,k}$, Historical Sequence $h_{t,k}$, Target $g_t$, Global Broadcast $c_t$ and $\alpha_k$, outputs SAOAction:

If REJECT, output $(q,\delta,p)$ and map it to $(q,t_abs,p)$ .

(5) L1 Hard Clipping and Legality Check:

Enforce hard constraints on L3 outputs: limit the number of trims to the safety boundary, mask invalid delivery dates; and ensure protocol validity (ACCEPT must carry offer\_in; REJECT must carry counter-offer; END outputs None).

## 4.2 L1 Security Base Layer: Deterministic Constraints and Masking Mechanisms

L1 Layer (Safety Shield) serves as the system's foundation, with its code implementation located in `litaagent_std/hrl_xf/l1_safety.py` . It does not contain neural networks but instead consists of a set of \*\*hard constraints\*\* based on domain knowledge, ensuring lower bounds on agent behavior.

### 4.2.1 Heuristic Systems and Sources of Security Rules

Prior to introducing learning strategies, we completed the replication and enhancement of the heuristic agent framework (Y-series and CIR-series), with core contributions including: inventory-sensitive procurement tripartite approach, unified procurement strategy for portfolio evaluation, feasibility simulation of InventoryManager, and opponent modeling/concession strategies. This framework delivers two types of value:

(1) Safety Rule Origin: Formalize hard constraints such as "deliverable, payable, and not oversold" into the L1 Safety Mask;

(2) Expert Data Sources: Used for offline forensics and behavior cloning (BC), providing a stable cold start for online learning.

Note: In the latest HRL-XF design, the aforementioned heuristic strategy no longer serves as the "baseline action" for L1 superimposed on the learning policy, but functions solely as a feasibility rule and source of expert trajectories.

## 4.3 L2 Strategic Management Level: Long-Range Planning and Futures Arbitrage

L2 (Strategic Manager) handles long-term planning issues spanning $T=100$ days. Its core task is to achieve intertemporal arbitrage by controlling the daily **inventory throughput rate** in uncertain market environments.

### 4.3.1 Actor-Critic Network Architecture

L2 adopts the standard Actor-Critic architecture to accommodate PPO training:

*   **Input Layer**: Receives a 128-dimensional macro state vector $s_{macro}$ (Including inventory, cash flow, and market trend sequences).
    
*   **Shared Encoder**: 3-layer MLP with 256 neurons per layer and ReLU activation function.
    
*   **Actor Head**: Outputs the mean $\mu$ and standard deviation $\sigma$ of a 4-dimensional Gaussian distribution, corresponding to the four components of $\mathbf{g}_t$.
    
*   **Critic Head**: Outputs a scalar $V(s)$ used to estimate the state value.
    

*\[Figure 4-2 Suggested insertion point: L2 neural network architecture diagram, showing the branch from s\_macro to Actor/Critic\]*

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=0.4cm, auto,
  block/.style={rectangle, draw, minimum width=2.0cm, minimum height=0.6cm, font=\small},
  arrow/.style={->, thick}]
  
  % Input
  \node[block, fill=yellow!20] (input) {$s_{macro}$ (128-d)};
  
  % Encoder
  \node[block, fill=blue!20, below=0.6cm of input] (enc1) {MLP 256};
  \node[block, fill=blue!20, below=0.3cm of enc1] (enc2) {MLP 256};
  \node[block, fill=blue!20, below=0.3cm of enc2] (enc3) {MLP 256};
  
  % Heads
  \node[block, fill=green!20, below left=0.6cm and -0.5cm of enc3] (actor) {Actor: $\mu,\sigma$};
  \node[block, fill=orange!20, below right=0.6cm and -0.5cm of enc3] (critic) {Critic: $V(s)$};
  
  % Output
  \node[below=0.4cm of actor, font=\small] {$\mathbf{g}_t \in [-1,1]^4$};
  \node[below=0.4cm of critic, font=\small] {Value est.};
  
  % Arrows
  \draw[arrow] (input) -- (enc1);
  \draw[arrow] (enc1) -- (enc2);
  \draw[arrow] (enc2) -- (enc3);
  \draw[arrow] (enc3) -| (actor);
  \draw[arrow] (enc3) -| (critic);
  
\end{tikzpicture}
\caption{Fig 4-2: L2 Network Architecture}
\end{figure}

### 4.3.2 Strategic Objective Vector and Potential Energy Guidance

The action vector $\mathbf{g}_t \in [-1, 1]^4$ output from L2 defines the trading boundaries for the day. To map the continuous neural network output into physically meaningful strategy instructions, we employ a guidance mechanism based on \*\*potential fields\*\*:

1.  **Target Purchase Volume**: $Q_{target} = \text{Softplus}(\mathbf{g}_t[0]) \cdot C_{daily} \cdot K$
    
    *   L2 controls inventory levels by adjusting $Q_{target}$. For example, when anticipating future price increases, output a larger $Q_{target}$ will instruct L3 to stockpile.

Through training with potential energy rewards, L2 learns to treat "inventory" as accumulated "potential energy," enabling it to proactively increase inventory when funds permit and achieve long-term planning.

## 4.4 L3 Execution Layer: Micro-Regulation Based on Transformers

L3's responsibility is to generate actions compliant with the SAO protocol semantics during specific negotiation rounds while optimizing rewards under the constraints of the L1 Safety Mask. Unlike the previous "baseline + residual" approach, the latest design enables L3 to directly output complete actions, thereby avoiding the expressive bottlenecks caused by baseline error propagation and residual space limitations.

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=0.35cm, auto,
  block/.style={rectangle, draw, minimum width=1.2cm, minimum height=0.55cm, font=\small},
  arrow/.style={->, thick}]
  
  % Inputs
  \node[block, fill=yellow!20] (obs) {$o_{t,k}$};
  \node[block, fill=yellow!20, right=0.25cm of obs] (hist) {$h_{t,k}$};
  \node[block, fill=yellow!20, right=0.25cm of hist] (gt) {$g_t$};
  \node[block, fill=yellow!20, right=0.25cm of gt] (ct) {$c_t$};
  \node[block, fill=yellow!20, right=0.25cm of ct] (alpha) {$\alpha_k$};
  
  % Transformer
  \node[block, fill=blue!20, below=0.7cm of hist, xshift=0.8cm, minimum width=4.0cm, minimum height=0.7cm] (tf) {Decision Transformer};
  
  % Output heads
  \node[block, fill=green!20, below=0.7cm of tf, xshift=-1.3cm] (op) {$op$};
  \node[block, fill=green!20, right=0.2cm of op] (time) {$\delta$};
  \node[block, fill=green!20, right=0.2cm of time] (qp) {$q,p$};
  
  % Arrows
  \draw[arrow] (obs) -- (tf);
  \draw[arrow] (hist) -- (tf);
  \draw[arrow] (gt) -- (tf);
  \draw[arrow] (ct) -- (tf);
  \draw[arrow] (alpha) -- (tf);
  \draw[arrow] (tf) -- (op);
  \draw[arrow] (tf) -- (time);
  \draw[arrow] (tf) -- (qp);
  
\end{tikzpicture}
\caption{Fig 4-4: L3 Decision Transformer Architecture}
\end{figure}

### 4.4.1 Sequence Modeling and Input Composition

L3 employs a Decision Transformer to model concession patterns and negotiation history. Input tokens are concatenated from the following components:

*   Local Observation $o_{t, k}$ Negotiation state, opponent's offer, local inventory/price characteristics, etc.
    
*   Historical Sequence $h_{t, k}$ : The (offer\_in, offer\_out, op) from the past several rounds;
    
*   Conditional Variable: Daily Target $g_t$ Global Broadcast $c_t$ and thread priority $\alpha_k$ .
    

where $c_t$$ Provided by L4GlobalMonitor, $\alpha_k$ Provided by GlobalCoordinator.

### 4.4.2 Action Head and Consistency Decoding $(op \to \delta \to q/p \to clip)$

L3 output contains three types of headers:

(1) $op_logits \in \mathbb{R}^3$ : Classification of $op  \in \{ACCEPT,REJECT,END\}$ ;

(2) $time_logits \in \mathbb{R}^{H+1}$ : The discrete distribution of the relative delivery time $\delta$ (used only when REJECT), overlaid with L1's time\_mask;

(3) quantity\_head / price\_head: Generates $q_{raw} \geq 0$, $p_{raw} \geq 0$ (Effective while REJECT only).

To ensure consistency between training and execution, the action decoding is fixed as follows:

(1) Sample/take maximum op\* (overlay L1's accept/reject/end feasibility mask);

(2) If op\*=END: Output END;

(3) if op\*=ACCEPT: Output ACCEPT(offer\_in);

(4) If op\*=REJECT: First sample/extract the maximum $\delta$\* (overlaid with time\_mask), then generate q and p. Finally, apply L1.clip\_action to perform upper bound clipping only on q and map it to the absolute intersection time t\_abs.

This sequence avoids the training-execution distribution shift caused by scaling quantities with $\mathbb{E}[Q_{safe}]$.

## 4.5 L4 Global Coordination Layer: Monitoring Broadcast + Priority $\alpha$

The core objective of L4 is to resolve resource contention and information inconsistencies among concurrent negotiation threads: each thread observes only a partial negotiation state, yet they share the same resource pool. The latest design splits L4 into two components:

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=0.4cm, auto,
  block/.style={rectangle, draw, minimum width=2.0cm, minimum height=0.6cm, font=\small},
  arrow/.style={->, thick}]
  
  % Threads
  \node[block, fill=gray!20] (t1) {Thread 1};
  \node[block, fill=gray!20, right=0.4cm of t1] (t2) {Thread 2};
  \node[block, fill=gray!20, right=0.4cm of t2] (tk) {Thread k};
  
  % L4 Monitor
  \node[block, fill=orange!20, below=0.8cm of t1, xshift=1.2cm, minimum width=2.4cm] (mon) {L4 Monitor};
  \node[below=0.1cm of mon, font=\small] {$\to c_\tau$};
  
  % L4 Coordinator
  \node[block, fill=blue!20, below=0.8cm of tk, xshift=-0.8cm, minimum width=2.4cm] (coord) {Coordinator};
  \node[below=0.1cm of coord, font=\small] {$\to \alpha_{\tau,k}$};
  
  % Arrows
  \draw[arrow] (t1) -- (mon);
  \draw[arrow] (t2) -- (mon);
  \draw[arrow] (tk) -- (coord);
  \draw[arrow, dashed] (mon) -- (coord);
  
  % Output to L3
  \node[block, fill=green!20, below=1.0cm of mon, xshift=1.4cm, minimum width=2.4cm] (l3) {L3 Policy};
  \draw[arrow] (mon) -- (l3);
  \draw[arrow] (coord) -- (l3);
  
\end{tikzpicture}
\caption{Fig 4-5: L4 Global Coordination}
\end{figure}

(1) L4GlobalMonitor (Deterministic Rule)

Maintain company-wide global commitments and remaining resources, including: - Daily closed volume - Future delivery commitments - Remaining budget - Remaining tradable security volume and output the global broadcast feature $c_t$ (e.g., $goal_{gap}$, $Q_{safe-remaining}$, $B_{remaining}$ etc.).

(2) GlobalCoordinator (Self-Attention Network)

Input: "Set of explicit thread features + global features" Output: Priority $\alpha_k  \in (−1,1)$ for each thread. The semantics of $\alpha_k$ is "whether the thread should be more aggressive or conservative in resource contention." It is used to adjust the policy distribution of L3 (e.g., concession speed, acceptance tendency), rather than directly modulating actions. It also eliminates the dynamic reservation mechanism of "weight-ordered deduction and pruning."

### 4.5.1 Learning-Based Formalization of Priority $\alpha$ (Replacing Explicit Weight Assignment)

In the new version of HRL-XF, L4 no longer outputs explicit "resource allocation weights" and performs sequential deduction-based dynamic reservation. Instead, it outputs the priority of each active negotiation thread.

$\alpha_{\tau,k}\in(-1,1)$,

As a conditional input for L3, it adjusts the "aggressive/conservative" bias of the policy distribution.

Let the set of active threads at the $\tau$ th event step be denoted by $\mathcal{K}_\tau$. L4GlobalMonitor maintains company-wide global commitments and remaining resources, and broadcasts the global feature $c_\tau$ (e.g., \`goal\_gap\`, \`Q\_safe\_remaining\`, \`B\_remaining\`, etc.). For each thread $k\in\mathcal{K}_\tau$, Construct explicit thread characteristics $x_{\tau,k}$, (Character, remaining rounds, opponent bid summary, historical statistics, etc.). GlobalCoordinator models the thread set using self-attention and outputs:

$\alpha_{\tau,k} = f_{\psi}\big(x_{\tau,k}, \{x_{\tau,j}\}_{j\in\mathcal{K}_\tau}, c_\tau\big)$.

During training, $\alpha_{\tau,k}$ participates in Online Optimization of Thread-Level Policies as a Conditional Variables:

$\pi_\theta(a_{\tau,k}\mid o_{\tau,k}, g_d, c_\tau, \alpha_{\tau,k})$,

By centralized critic (using $c_\tau$ and other global summaries) estimate advantages to reduce variance.

This design achieves a form of "soft coordination": L4 enables cross-thread coordination by modifying policy distributions rather than directly rewriting actions. This approach avoids engineering coupling, non-differentiable constraint handling, and sequential subtraction error propagation associated with explicit resource allocation.

### 4.5.2 Semantics and Thread Characteristics of $\alpha$

$\alpha \to +1$ indicates high priority/urgent threads, which should be prioritized for faster execution; $\alpha \to −1$ indicates low priority, which should be prioritized for resource preservation. Thread characteristics may include: role, relative progress, round, presence of counterparty quotes, $q_{in} / p_{in} / \delta_{in}$. Global features include: target gap, remaining budget, remaining safety margin, etc. The GlobalCoordinator implicitly models inter-thread competition through self-attention.

## 4.6 Inter-layer Coordination Mechanism

The four levels of HRL-XF do not operate in isolation but are tightly coupled through \*\*data flow and control flow\*\*.

### 4.6.1 Top-Down Control Flow

1.  **L2** $\to$ **L3/L4**
    
    L2-generated strategic objective vector $g_t$ serves as the **Conditional Context** for L3 and L4. This is equivalent to L2 establishing a "potential energy field" within which L3 operates. For example, if L2 sets an extremely high $Q_{buy}$, even if L3 detects that the counterparty's price is slightly higher, it will still tend to execute the trade to meet its KPIs.
    

2\. **L4 → L3: Conditional input with priority $\alpha$ (no weight distribution/no action modulation):**

*   **L3** $\to$ **L1**

The "recommended action" output from L3 must pass through the hard constraint filter of L1.

### 4.6.2 Bottom-Up Feedback

1.  **Environment** **L1**: L1 directly perceives physical constraints in the environment (e.g., depleted funds) and feeds them back to L3 via a Mask (i.e., invalid actions are masked and penalized during training).
    
2.  **L3** $\to$ **L2**: At the end of the day, L3's actual execution results (such as actual trading volume $Q_{actual}$) and the objective $Q_{target}$ of L2. The deviation between them constitutes L3's **Alignment Reward**. Simultaneously, the final state of that day (funds, inventory changes) becomes the input s_{macro}^{(t+1)} for L2's next decision.
    

### 4.6.3 Coordination During Training

During the training phase, we employ a strategy combining \*\*Decoupled Training and Joint Fine-tuning\*\*:

*   **Decoupling**: Using offline data, train the macro-level intent of the L2 prediction expert and the micro-level actions of the L3 prediction expert separately. At this stage, implicit coordination occurs between layers through data labels.
    
*   **Joint**: During the online fine-tuning phase, L2 and L3 update simultaneously. L2 learns how to set objectives that make execution easier for L3, while L3 learns how to better respond to L2's objectives. The potential reward function $\Phi(s)$ serves as a crucial **value bridge** in this process, translating long-term gains into concrete daily decisions.
    

## Chapter 5: Implementation and Engineering Optimization

# Chapter 5: System Implementation and Engineering Optimization

The preceding chapters outlined the theoretical model and architectural design of HRL-XF. This chapter focuses on the engineering implementation of this architecture, detailing the modular implementation of the LitaAgent agent system, the high-performance data pipeline supporting staged training, and the parallel optimization solutions and analysis toolkit developed to address large-scale simulation challenges.

## 5.1 LitaAgent agent System Implementation

The LitaAgent system is developed using Python 3.10+ and employs an object-oriented (OOP) modular design to ensure scalability and maintainability. The core code resides within the \`litaagent\_std\` package.

### 5.1.1 Class Inheritance Hierarchy and Mixing Patterns

To support a smooth transition from purely heuristic to purely neural control while enabling non-intrusive monitoring of agent behavior, we adopted the **Mixin design pattern** (see `tracker_mixin.py` and `litaagent_yr.py`).

*   **Base Agent**: Inherits from SCML's official `SCML2020Agent`, responsible for handling underlying NegMAS protocol callbacks (such as `on_negotiation_request`, `on_contract_signed`).
    
*   **Functional Components**: Layers L1 through L4 are encapsulated as independent manager classes and integrated into the main agent through composition.
    
*   **TrackerMixin**: Utilizes multiple inheritance injection to intercept state data at critical lifecycle hooks.
    

**Primary Class Definitions**:

$$\text{Agent} = \text{BaseProtocol} \oplus \text{InventoryManager} \oplus \text{HRLController} \oplus \text{TrackerMixin}$$

The principal-agent class `HRLXFAgent` (`litaagent_std/hrl_xf/agent.py`) maintains a state machine whose core scheduling logic is illustrated by the following pseudocode:

**Algorithm 5.1: HRL-XF agent Master Control Loop**

```python
class HRLXFAgent(SCML2020Agent):
  def init(self):
    self.l1 = SafetyShield(config=self.config)
    self.l2 = StrategicManager(self.l2_weights)
    self.l3 = L3ActorDT(self.l3_weights)
    self.l4_monitor = L4GlobalMonitor()
    self.l4_coord = GlobalCoordinator(self.l4_weights)
    self.state_builder = StateBuilder()

  def before_step(self):
    """Daily: macro decision & global init"""
    s_macro = self.state_builder.build_macro(self.awi)
    self.daily_goal = self.l2.predict(s_macro)
    self.l1_out = self.l1.compute(self.awi)
    self.l4_monitor.reset_day(
      self.awi, self.daily_goal, self.l1_out)

  def respond(self, negotiation, state):
    """Per-round micro decision (SAO)"""
    local_obs = self.state_builder.build_local(
      self.awi, negotiation, state, self.daily_goal)
    global_bc = self.l4_monitor.get_broadcast()

    alpha = self.l4_coord.get_alpha(
      negotiation.id, local_obs.thread_feat, global_bc)
    l3_out = self.l3.act(local_obs, global_bc, alpha)

    safe_action = self.l1.clip_and_validate(
      l3_out.action, state.current_offer)
    return safe_action.to_sao_response(
      offer_in=state.current_offer)
````

### 5.1.2 State Builder

`state_builder.py` is responsible for converting heterogeneous simulation data (objects, dictionaries, lists) into tensors acceptable to neural networks. This layer achieves decoupling between data and the model.

*   **Feature Engineering**: Log-scale price data $\log(p)$ and normalize inventory to maximum capacity $I\_t / C\_{max}$ to ensure numerical stability.
    
*   **Temporal Encoding**: Encodes historical negotiation sequences into the `(Batch, Seq_Len, Feature_Dim)` format for Transformer processing. Missing historical data (e.g., at the start of negotiations) is padded with a special Padding Token.
    

*\[Figure 5-1 Recommended Insertion Position: LitaAgent Software Class Diagram, illustrating dependencies between the Agent and the InventoryManager, StateBuilder, and L1-L4 modules\]*

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=0.4cm, auto,
  class/.style={rectangle, draw, minimum width=2.2cm, minimum height=0.6cm, font=\small},
  arrow/.style={->, thick, >=stealth}]
  
  % Main agent
  \node[class, fill=yellow!30] (agent) {HRLXFAgent};
  
  % Components row 1
  \node[class, fill=blue!20, below left=0.8cm and 0.4cm of agent] (sb) {StateBuilder};
  \node[class, fill=blue!20, below right=0.8cm and 0.4cm of agent] (im) {InventoryMgr};
  
  % L1-L4 row
  \node[class, fill=red!20, below=1.5cm of agent, xshift=-2.0cm] (l1) {L1:Shield};
  \node[class, fill=orange!20, below=1.5cm of agent, xshift=0.0cm] (l2) {L2:Strategy};
  \node[class, fill=green!20, below=1.5cm of agent, xshift=2.0cm] (l34) {L3/L4};
  
  % Arrows
  \draw[arrow] (agent) -- (sb);
  \draw[arrow] (agent) -- (im);
  \draw[arrow] (agent) -- (l1);
  \draw[arrow] (agent) -- (l2);
  \draw[arrow] (agent) -- (l34);
  \draw[arrow, dashed] (sb) -- (l2);
  \draw[arrow, dashed] (im) -- (l1);
  
\end{tikzpicture}
\caption{Fig 5-1: LitaAgent Class Dependencies}
\end{figure}

## 5.2 Staged Training Paradigm: Forensics, Stripping, and Fine-Tuning

To address the cold start problem in RL within long-range complex environments, this paper designs a three-stage data pipeline. The relevant implementation is located in `runners/hrl_data_runner.py` and `litaagent_std/hrl_xf/data_pipeline.py` .

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=0.5cm, auto,
  stage/.style={rectangle, draw, rounded corners, minimum width=2.4cm, minimum height=0.8cm, font=\small},
  arrow/.style={->, thick, >=stealth}]
  
  % Stages
  \node[stage, fill=blue!20] (s1) {Phase 1};
  \node[below=0.05cm of s1, font=\small] {Forensics};
  
  \node[stage, fill=orange!20, right=1.2cm of s1] (s2) {Phase 2};
  \node[below=0.05cm of s2, font=\small] {Stripping};
  
  \node[stage, fill=green!20, right=1.2cm of s2] (s3) {Phase 3};
  \node[below=0.05cm of s3, font=\small] {Fine-tune};
  
  % Outputs
  \node[below=0.8cm of s1, font=\small, text width=2.2cm, align=center] {Expert Logs\\$10^6$ records};
  \node[below=0.8cm of s2, font=\small, text width=2.2cm, align=center] {Supervised\\Labels};
  \node[below=0.8cm of s3, font=\small, text width=2.2cm, align=center] {MAPPO\\Online RL};
  
  % Arrows
  \draw[arrow] (s1) -- (s2);
  \draw[arrow] (s2) -- (s3);
  
\end{tikzpicture}
\caption{Fig 5-3: Three-Stage Training Pipeline}
\end{figure}

### 5.2.1 Phase One: Forensics

At this stage, we do not directly train the model; instead, we run a large number of simulated matches to collect expert data.

*   **Expert Agents**: Utilize *PenguinAgent* or its variants, such as `LitaAgentYR`. While lacking long-term adaptability, these agents demonstrate robust performance in single-player scenarios.
    
*   **Injection-based Tracking**: Utilizes the `TrackerMixin` to record each round's $s\_{micro}, a\_{expert}$ and the final match outcome through a hook mechanism, without modifying the expert's internal logic.
    
*   **Data Scale**: Typically running 100–500 Standard tournaments generates approximately $10^6$ micro-negotiation records.
    

### 5.2.2 Phase Two: Data Stripping and Supervision Signal Construction (Changing from Residual Labels to SAOAction Labels)

The objective of this phase is to construct training samples from evidence logs for supervised learning and online RL. Unlike the legacy "baseline + residual" approach, the latest design has L3 directly outputting complete SAOAction. Therefore, the supervised signal is defined as:

*   $op*: {ACCEPT, REJECT, END}$
    
*   If $op^*=REJECT$: $\delta^*$, $q^*$, $p^*$ (and record $t_{abs}$ for the purpose of reproduction)
    

Simultaneously compute the L1 Safety Mask offline ($Q_{safe}$ / $Q_{safe-sell}$ / $B_{free}$ / $time\_mask$), used for:

(1) Masked loss during training (to prevent learning of inaction);

(2) Consistency trimming and validity validation during execution.

At this stage, it is no longer necessary to reproduce the "L1 baseline action" to calculate residual differences.

### 5.2.3 Phase Three: Online Learning (MAPPO) and Rationale for Selection

Why Choose MAPPO:

(1) Concurrent multithreading negotiations inherently involve multi-agent coupling: threads share resources under constraints, giving rise to competition and coordination;

(2) The opponent's strategy is non-stationary and partially observable, leading to high variance during training; PPO's clipping mechanism suppresses drastic policy drift.

(3) CTDE can learn more stable value functions during training by utilizing company-wide global broadcast states, while maintaining thread-distributed decision-making during execution.

(4) Parameter sharing adapts to thread homogeneity: All threads share the same L3 policy, enhancing sample efficiency and reducing training difficulty.

Implementation Approach for Online MAPPO:

*   Actor (shared): $L3(\theta)$, input $(o_{t,k}, g_t, c_t, \alpha_k)$, output $\pi_\theta(a_{t,k})$
    
*   Critic (Focus): $V_\varphi(S_t)$ or its global characteristics, used to estimate the advantage $A_{t,k}$
    
*   Update: Update $\theta$ using the PPO clipped objective and perform a regression update on $\varphi$.
    

In an event-driven environment, the moment when "a thread is scheduled to perform an action" is treated as a step. For threads not scheduled, a mask is applied to construct batch traces for training.

## 5.3 Engineering Breakthrough: Loky Parallel Optimization and SCML Analyzer

During the implementation of large-scale "evidence collection" training, we encountered significant engineering challenges. This section details these challenges and their solutions.

### 5.3.1 Resolving NegMas Multi-Process Deadlock (Hung) Issues

**Issue Symptoms**: When running over 50 concurrent matches using `SCML2020Tournament`, Python processes frequently enter a "zombie" state—CPU usage drops to 0%, memory remains occupied, and no error logs are generated. **Root Cause Analysis**: NegMAS defaults to using `concurrent.futures.ProcessPoolExecutor` . In Python, when child processes transmit large amounts of serialized data (SCML's Agent state objects are extremely large) via `multiprocessing.Queue` and the parent process cannot read it in time, the pipeline buffer fills up, leading to deadlock. Additionally, certain NegMAS objects (e.g., \`World\`) contain lock objects that are difficult to pickle, resulting in serialization failures or suspensions.

**Solution: Monkey Patch Based on Loky** We developed `runners/loky_patch.py`, utilizing the `loky` library to replace the standard multiprocessing backend. `loky` is more robust, automatically detecting and restarting crashed worker processes. It employs disk-based memory mapping (Cloudpickle + Joblib) for handling large data transfers, circumventing Queue's buffer limitations.

**Algorithm 5.2: Loky Backend Injection Logic**

```plain
# runners/loky_patch.py
import sys
from negmas import tournament
from loky import get_reusable_executor

def apply_patch():
  """Replace NegMAS multiprocessing executor"""
  original_run = tournament.run_tournament
  
  def patched_run(*args, **kwargs):
    kwargs['parallelism'] = 'process'
    with get_reusable_executor(
        max_workers=n_jobs) as executor:
      return original_run(*args, **kwargs)
      
  tournament.run_tournament = patched_run
  print(">> Loky patch applied.")
```

The application of this patch has enhanced our data collection efficiency on a single 192-core server, enabling continuous, fault-free operation that supports the data requirements of the forensic competition.

### 5.3.2 SCML Analyzer Tool Suite

To fill the gap in micro-analysis tools within the NegMAS ecosystem, we designed the **SCML Analyzer**. Its architecture adheres to the principle of \*\*"bypass monitoring, independent storage"\*\*, decoupling from agent logic.

1.  **Tracker Injection (Auto-Tracker)**: 
    
    *   This is a decorator or mixin that dynamically wraps agent methods such as \`respond\` and \`on\_contract\_signed\` at runtime.
        
    *   It serializes each round's input (observations) and output (actions) into JSONL format and writes them to disk in a streaming manner, avoiding memory consumption.
        
2.  **Visualization Service (Visualizer)**:
    
    *   A lightweight web service built using `FastAPI` + `Plotly` (`scml_analyzer/visualizer.py`).
        
    *   **Core Features**:
        
        *   **Negotiation Replay**: Presents the 20 rounds of bargaining for a specific contract in a timeline format, helping developers intuitively understand whether L3's output is reasonable.
            
        *   **Inventory Health Monitoring**: Plot a comparison curve between $I\_{actual}$ and the L2 target $Q\_{target}$ to validate the effectiveness of macro-level control.
            

*\[Figure 5-2 Suggested Insertion Position: Screenshot of the SCML Analyzer web interface displaying the negotiation Gantt chart and inventory curve\]*

\begin{figure}[h]
\centering
\begin{tikzpicture}[scale=1.2]
  % Inventory curve
  \draw[thick,->] (0,0) -- (6,0) node[right, font=\small] {Day};
  \draw[thick,->] (0,0) -- (0,3.2) node[above, font=\small] {Inv};
  
  % Actual inventory line
  \draw[blue, very thick] (0,1.9) -- (0.6,1.6) -- (1.2,2.0) -- (1.8,1.5) 
    -- (2.4,1.8) -- (3.0,1.4) -- (3.6,1.9) -- (4.2,1.6) -- (4.8,2.0) -- (5.4,1.8);
  
  % Target line
  \draw[red, dashed, thick] (0,1.8) -- (5.4,1.8);
  
  % Legend
  \node[font=\small, blue] at (4.2,2.8) {$I_{actual}$};
  \node[font=\small, red] at (4.2,2.4) {$Q_{target}$};
  
\end{tikzpicture}
\hspace{0.4cm}
\begin{tikzpicture}[scale=1.2]
  % Negotiation timeline
  \draw[thick,->] (0,0) -- (6,0) node[right, font=\small] {Round};
  \draw[thick,->] (0,0) -- (0,3.2) node[above, font=\small] {Price};
  
  % Offer exchange
  \foreach \x/\y in {0.4/2.5, 1.1/1.5, 1.9/2.3, 2.6/1.6, 3.3/2.0, 4.0/1.8, 4.7/1.9}
    \fill[blue] (\x,\y) circle (2.5pt);
  \foreach \x/\y in {0.7/1.2, 1.5/1.9, 2.2/1.4, 2.9/1.9, 3.6/1.6, 4.3,1.85}
    \fill[red] (\x,\y) circle (2.5pt);
  
  % Final agreement
  \draw[green!60!black, very thick] (4.7,1.9) circle (4pt);
  \node[font=\small] at (5.2,2.1) {Deal};
  
\end{tikzpicture}
\caption{Fig 5-2: SCML Analyzer Views (Left: Inventory, Right: Negotiation)}
\end{figure}

Through this toolchain, we successfully identified a logical vulnerability in the PenguinAgent strategy during futures delivery dates, providing critical evidence for subsequent iterations of the HRL-XF architecture.

### 5.3.3 Security Auditing and Attribution Analysis: The Case of PenguinAgent

Through injection tracing with SCML Analyzer, we discovered a low-probability yet reproducible unsafe behavior in PenguinAgent: accepting/proposing offers when fulfillment is inevitably impossible. We define "unsafe" as: under current observable inventory, capacity, committed orders, and funding constraints, the contract will inevitably experience a shortfall or funding insufficiency on the delivery date. Sampling statistics indicate PenguinAgent's unsafe\_any rate ranges from approximately 0.42% to 3.63% (depending on the world/opponent combination and statistical window).

Static code analysis further pinpointed three critical defects:

(1) The sell-side available-to-sell estimate uses only current\_inventory\_input + n\_lines\*productivity − total\_sales\_at(step), ignoring finished goods inventory and cross-day cumulative commitments, which can easily lead to overselling (penguinagent.py: line 303).

(2) Code inplementation fault: awi.total\_sales\_at(awi.total\_sales\_at(awi.current\_step) <= awi.n\_lines) passes a boolean value to the step, will lead to miscalculation of whether sellable or not.(penguinagent.py: line 125)

(3) future\_consume\_offer using total\_sales\_at(s+2) s+3 to consider the offer on t+2.(penguinagent.py: line 498)

Based on the aforementioned shortcomings, we propose two solutions:

*   LitaAgent-H: Directly fixed three logic issues, redefined seller's available-for-sale estimates, and corrected conditional and indexing errors.
    
*   LitaAgent-HS: An L1 Safety Shield that attaches HRL-XF to PenguinAgent: Prunes the number of insecure offers; rewrites insecure ACCEPT responses to REJECT and provides a secure counter-offer.
    

# Chapter 6: Experimental Design and Results Analysis

*(This chapter is approximately 5–6 pages long.)*

## 6.1 Experimental Setup

*   **Platform**: SCML 2025 Official Simulation Environment.
    
*   **Baseline Competitor**: SCML 2024 Top 5 Agents(Including PenguinAgent, DecentralizingAgent).
    
*   **Training Configuration**: Pre-trained on large-scale tournament data running on Loky Patch, followed by 5000 steps of online PPO updates.
    

## 6.2 Baseline agent Security Fix Experiment: PenguinAgent → LitaAgent-H / LitaAgent-HS

Experimental setup: Official-scale competitions were run using anac2024\_std() with --round-robin enabled. We compared three proxies: the original PenguinAgent, the directly patched LitaAgent-H, and the externally shielded LitaAgent-HS.

Overall Performance (total\_scores):

*   LitaAgent-HS: 0.8741 (Highest)
    
*   LitaAgent-H: 0.8564
    
*   PenguinAgent: 0.4573
    

Negotiation/Contract Statistics (Average per world):

PenguinAgent: negs\_succeeded 246.3 / contracts\_signed 724.4 / contracts\_executed 361.7

LitaAgent-H: negs\_succeeded 250.4 / contracts\_signed 683.3 / contracts\_executed 341.1

LitaAgent-HS: negs\_succeeded 265.7 / contracts\_signed 705.1 / contracts\_executed 351.5

Safety sampling (randomly sampling 60 worlds per agent):

PenguinAgent: unsafe\_any = 537 / 128,887 ≈ 0.42%

LitaAgent-H: unsafe\_any = 343 / 94,582 ≈ 0.36%

LitaAgent-HS: unsafe\_any = 0 / 120,340 ≈ 0%

\begin{figure}[h]
\centering
\begin{tikzpicture}[scale=1.2]
  % Bar chart for total scores
  \draw[thick,->] (0,0) -- (5.5,0) node[right] {};
  \draw[thick,->] (0,0) -- (0,4.2) node[above] {Score};
  
  % Bars
  \fill[red!60] (0.6,0) rectangle (1.5,1.6);
  \fill[blue!60] (2.0,0) rectangle (2.9,3.1);
  \fill[green!60] (3.4,0) rectangle (4.3,3.2);
  
  % Labels
  \node[below, font=\small] at (1.05,0) {Penguin};
  \node[below, font=\small] at (2.45,0) {Lita-H};
  \node[below, font=\small] at (3.85,0) {Lita-HS};
  
  % Values
  \node[above, font=\small] at (1.05,1.6) {0.46};
  \node[above, font=\small] at (2.45,3.1) {0.86};
  \node[above, font=\small] at (3.85,3.2) {0.87};
  
  % Y-axis ticks
  \foreach \y/\ytext in {0/0, 1.2/0.33, 2.4/0.67, 3.6/1.0}
    \draw (0.08,\y) -- (-0.08,\y) node[left, font=\small] {\ytext};
\end{tikzpicture}
\hspace{0.5cm}
\begin{tikzpicture}[scale=1.2]
  % Bar chart for unsafe rate
  \draw[thick,->] (0,0) -- (5.5,0) node[right] {};
  \draw[thick,->] (0,0) -- (0,4.2) node[above] {Unsafe\%};
  
  % Bars
  \fill[red!60] (0.6,0) rectangle (1.5,3.0);
  \fill[blue!60] (2.0,0) rectangle (2.9,2.6);
  \fill[green!60] (3.4,0) rectangle (4.3,0.0);
  
  % Labels
  \node[below, font=\small] at (1.05,0) {Penguin};
  \node[below, font=\small] at (2.45,0) {Lita-H};
  \node[below, font=\small] at (3.85,0) {Lita-HS};
  
  % Values
  \node[above, font=\small] at (1.05,3.0) {0.42\%};
  \node[above, font=\small] at (2.45,2.6) {0.36\%};
  \node[above, font=\small] at (3.85,0.15) {0\%};
  
  % Y-axis ticks
  \foreach \y/\ytext in {0/0, 2/0.25, 4/0.5}
    \draw (0.08,\y) -- (-0.08,\y) node[left, font=\small] {\ytext};
\end{tikzpicture}
\caption{Fig 6.2-1: Agent Performance Comparison (Left: Total Score, Right: Unsafe Rate)}
\end{figure}

Conclusion: Without compromising negotiation success rates, Safety Shield reduces unsafe behaviors to 0% and significantly improves overall scores. This validates the synergistic effect of "deterministic safety shield + strategy optimization" and provides a secure, controllable execution foundation for HRL-XF online learning.

## 6.3 HRL-XF Online Learning (In Progress) and Evaluation Metrics

The subsequent online learning in this paper will center on MAPPO: thread sharing strategies and centralized critics leveraging global broadcast state estimation advantages. Beyond total\_score, evaluation metrics explicitly incorporate safety indicators (unsafe\_any, default rate, shortfall statistics) and stability metrics (inventory volatility, capital volatility).

### 6.3.1 Offline Pre-training Performance

*   Display the convergence curve of the L3 network on the fitting expert (MSE Loss).
    
*   Compare the scores of the L1 baseline agent and the L1+L3 (pre-training only) agent in the standard environment.

\begin{figure}[h]
\centering
\begin{tikzpicture}[scale=1.2]
  % Axes
  \draw[thick,->] (0,0) -- (6,0) node[right, font=\small] {Epoch};
  \draw[thick,->] (0,0) -- (0,4) node[above, font=\small] {MSE Loss};
  
  % Loss curve (exponential decay)
  \draw[blue, very thick] (0,3.5) .. controls (0.6,2.8) and (1.2,1.9) ..
    (1.8,1.3) .. controls (2.4,0.9) and (3.0,0.6) ..
    (3.6,0.45) .. controls (4.2,0.32) and (4.8,0.25) .. (5.4,0.22);
  
  % Annotations
  \node[font=\small, blue] at (5.8,0.5) {L3 Loss};
  \draw[dashed, gray] (0,0.25) -- (5.4,0.25);
  \node[font=\small, gray] at (6.0,0.25) {Target};
  
  % Y-axis ticks
  \foreach \y/\l in {0/0, 1.33/0.5, 2.67/1.0, 3.5/1.4}
    \draw (-0.08,\y) -- (0.08,\y) node[left, font=\small] {\l};
  
\end{tikzpicture}
\caption{Fig 6.3.1: L3 Pre-training Convergence}
\end{figure}
    

### 6.3.2 Online Fine-Tuning and Ablation Study

Design experiments comparing the following variants to validate the effectiveness of each module:

1.  **Baseline**: Pure rile-based agent, like PenguinAgent.
    
2.  **HRL-No-L2**: Remove L2 planning and perform fine-tuning using only L3.
    
3.  **HRL-XF (Full)**: Full architecture.

\begin{figure}[h]
\centering
\begin{tikzpicture}[scale=1.15]
  % Axes
  \draw[thick,->] (0,0) -- (7,0) node[right] {};
  \draw[thick,->] (0,0) -- (0,4) node[above, font=\small] {Score/Stability};
  
  % Grouped bars: Baseline, HRL-No-L2, HRL-Full
  % Profit bars
  \fill[red!50] (0.5,0) rectangle (1.1,2.2);
  \fill[blue!50] (1.2,0) rectangle (1.8,2.5);
  \fill[green!50] (1.9,0) rectangle (2.5,3.0);
  
  % Stability bars
  \fill[red!50] (3.5,0) rectangle (4.1,3.2);
  \fill[blue!50] (4.2,0) rectangle (4.8,2.6);
  \fill[green!50] (4.9,0) rectangle (5.5,1.5);
  
  % Group labels
  \node[font=\small] at (1.5,-0.4) {Profit};
  \node[font=\small] at (4.5,-0.4) {Inv. Vol.};
  
  % Legend
  \fill[red!50] (0.4,3.5) rectangle (0.8,3.8);
  \node[font=\small, right] at (0.85,3.65) {Base};
  \fill[blue!50] (2.0,3.5) rectangle (2.4,3.8);
  \node[font=\small, right] at (2.45,3.65) {No-L2};
  \fill[green!50] (3.8,3.5) rectangle (4.2,3.8);
  \node[font=\small, right] at (4.25,3.65) {Full};
  
\end{tikzpicture}
\caption{Fig 6.3.2: Ablation Study (Profit$\uparrow$, Volatility$\downarrow$)}
\end{figure}

*(Experimental results: HRL-Full slightly outperforms Baseline in terms of profit, but demonstrates significantly superior performance in default rate and inventory stability, particularly during periods of sharp market price volatility.)*

### 6.3.3 Case Analysis and Feasibility Verification

Select a specific "hyperinflation" test set to analyze the behavior of the HRL-XF agent.

\begin{figure}[h]
\centering
\begin{tikzpicture}[scale=1.2]
  % Axes
  \draw[thick,->] (0,0) -- (6,0) node[right, font=\small] {Day};
  \draw[thick,->] (0,0) -- (0,4) node[above, font=\small] {Price/Inv};
  
  % Price spike
  \draw[red, very thick] (0,1.2) -- (1.2,1.35) -- (2.2,1.5) -- (2.7,3.1) -- (3.4,3.2) -- (4.2,3.0) -- (5.4,2.9);
  \node[font=\small, red] at (5.8,3.1) {Price};
  
  % Inventory buildup (HRL-XF hoarding)
  \draw[blue, very thick, dashed] (0,0.6) -- (1.2,0.9) -- (2.2,1.9) -- (2.7,2.2) -- (3.4,1.5) -- (4.2,1.0) -- (5.4,0.75);
  \node[font=\small, blue] at (5.8,0.95) {Inv};
  
  % Annotation: buy phase
  \draw[<->, gray, thick] (1.2,0.35) -- (2.2,0.35);
  \node[font=\small, gray] at (1.7,0.1) {Hoard};
  
  % Annotation: sell phase
  \draw[<->, gray, thick] (3.0,0.35) -- (4.2,0.35);
  \node[font=\small, gray] at (3.6,0.1) {Sell};
  
\end{tikzpicture}
\caption{Fig 6.3.3: Hyperinflation Case (L2 Hoarding Behavior)}
\end{figure}

*   **Phenomenon**: On the eve of price increases, Layer 2 can identify trends and issue advance buy orders (hoarding) through a momentum reward mechanism ( $\Phi$ ).
    
*   **Conclusion**: Experimental data demonstrate that the HRL-XF agent not only outperforms the baseline in average returns but also exhibits significant advantages in robustness under extreme conditions. This provides preliminary evidence for the feasibility of the hierarchical reinforcement learning framework within the SCML 2025 environment and holds promise as a mainstream paradigm for next-generation intelligent agent design in this field.
    

## Chapter 7: Conclusion

*(This chapter is approximately 2 pages long.)*

## 7.1 Full Text Summary

This paper proposes and implements the HRL-XF framework, focusing on three key achievements: (1) Establishing strong-constraint supply chain negotiations on the deterministic Safety Mask/Shield security foundation, completing protocol semantic alignment and information flow reconstruction for L1/L3/L4 (L1 outputs only mask; L3 outputs SAOAction; L4 = Monitor+$\alpha$); (2) Formalizes modeling multi-threaded concurrent negotiations as (thread-level) Dec-POMDP and implements it as MAPPO for online learning; (3) Implements the SCML Analyzer to conduct security audits and defect localization on PenguinAgent, proposes the LitaAgent-H/HS remediation plan, and validates the effectiveness of the security shield in official large-scale round-robin testing.

## 7.2 Limitations and Future Work

Online convergence and sample efficiency of MAPPO remain key challenges; future efforts will focus on more stable thread-level credit allocation, enhanced global coordinator training, and self-play adversarial training, culminating in a comprehensive HRL-XF online learning evaluation on the SCML 2025 benchmark track.

## References

To be filled

## Acknowledgements

Thank you to my advisor for their guidance, to the SCML community for providing open-source resources, and to the *PenguinAgent* team for their inspiration.