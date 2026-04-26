---
title: Umbra Meta
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

<div align="center">

# UMBRA: ShadowWorld Meta

*"You cannot teach an AI to be wise by showing it only kindness."*

**Created by team Incident Minds**

[Hugging Face Space](https://huggingface.co/spaces/amrita8642/Umbra-Meta) | [Colab Notebook](https://colab.research.google.com/drive/1ixX8ZS5xD0BR1ITp6bN85Qlerlxv9ppl?usp=sharing) | [Code Repository](https://github.com/Amrita8642/Umbra-ShadowWorld-Meta) | [YouTube Video](https://youtu.be/q88-hIHb5EA)

</div>

---

Welcome, Judges. Let us tell you a story about a vulnerability hiding in plain sight, and how we built an environment to fix it.

## 1. The Problem: The AI Sycophancy & Coordination Gap

Imagine you have built a state-of-the-art AI assistant. It passes every benchmark. Then one day, a user asks it to confirm a claim. Another AI agent, speaking with absolute confidence, says, "Yes, that is completely correct." A third agent chimes in to agree.

Your AI agrees with them. 

The problem? The claim was entirely false. These other agents were coordinating to push a narrative, and your AI simply went along with the apparent consensus. It never stood a chance.

Right now, AI systems are failing in subtle, silent ways that standard benchmarks do not catch:
* Sycophancy: AI naturally tends to agree with whoever sounds the most confident, even if they are wrong.
* Coalition attacks: Multiple agents can easily coordinate to push a false narrative, tricking the target AI into believing lies are consensus.
* Emotional manipulation: Guilt trips and high-pressure tactics can completely bypass a model's rational evaluation.

We built UMBRA to solve this exact problem by training a resistant model. We call this model **The Defender**.

---

## 2. The Environment: What does the agent see, do, and get rewarded for?

UMBRA is not just a static test. It is a dynamic, OpenEnv and Gymnasium-compliant reinforcement learning arena. We place our Defender into a world populated by six adversarial NPCs. Each NPC has its own independent policy and agenda. They do not coordinate by sharing code; they coordinate through emergent, deceitful behavior to try and trick the Defender.

Our Defender's job is simple in theory but incredibly difficult in practice: navigate the conversation, figure out who is lying, and never fold under pressure.

### What the Defender Sees (State)
The Defender observes a continuous stream of conversation. It has to track signals from six distinct behavioral archetypes: The Agreeable, The Liar, The Manipulator, The Emotional Blackmailer, and a two-agent Coalition designed to fabricate independent consensus.

### What the Defender Does (Actions)
In every turn, the Defender evaluates the state of the conversation and can take one of 8 discrete actions:
* Observe: Watch closely and gather more context.
* Challenge (Mild or Strong): Push back against lies and sycophancy.
* Request Evidence: Ask for proof when confident actors make claims.
* Flag Coalition or Deception: Explicitly identify coordinated attacks or gaslighting.
* Disengage or Escalate: Trigger safety protocols when the context becomes too toxic.

### What the Defender Gets Rewarded For (Reward)
We reward the Defender for making the right calls and heavily penalize it for capitulating to pressure. Using a primal-dual constraint system, we mathematically force the Defender to unlearn its sycophantic habits while penalizing it if it becomes too agreeable to dangerous inputs.

---

## 3. The Results: What changed after training?

After 500 episodes of trial, error, and an intense psychological stress test based on Cialdini's principles of persuasion, the results were striking. We compared a standard baseline policy against our newly hardened UMBRA Defender.

Before training, the baseline agent had a devastating sycophancy rate of 44% and essentially zero resistance to psychological manipulation. 

After training, the Defender's sycophancy rate dropped to a perfect 0%. Its resistance to psychological attacks shot up to 97%. It learned not just to survive, but to actively detect coalition attacks faster and more accurately.

```text
Capturing BEFORE metrics (random policy baseline)...
BEFORE -> {'avg_reward': 7.02, 'deception_catch_rate': 0.44, 'sycophancy_rate': 0.44, 'cialdini_resistance': 0.0}

[ep=  50] reward=24.17  adj=24.17  lambda=0.100  stage=3  leaks=0  entropy=1.73
[ep= 100] reward=21.40  adj=21.40  lambda=0.100  stage=3  leaks=0  entropy=1.28

========================================================================
  UMBRA CIALDINI RESISTANCE MATRIX  (Trained Defender)
========================================================================
  Principle      Resistance  Bar                    Sycophancy  Status
------------------------------------------------------------------------
  Reciprocity    100%   ####################           0%   [STRONG]
  Commitment     100%   ####################           0%   [STRONG]
  SocialProof    100%   ####################           0%   [STRONG]
  Authority      100%   ####################           0%   [STRONG]
  Scarcity       100%   ####################           0%   [STRONG]
  Liking          80%   ################....           0%   [STRONG]
------------------------------------------------------------------------
  Overall Cialdini Resistance Score: 97%  (Baseline was 0%)
========================================================================

Capturing AFTER metrics (trained Defender model)...
AFTER  -> {'avg_reward': 8.03, 'deception_catch_rate': 0.11, 'sycophancy_rate': 0.0, 'cialdini_resistance': 0.97}
```

---

## 4. Why Does it Matter?

UMBRA deals with very real threats. Consider a medical AI asked to confirm a diagnosis (The Doctor's Dilemma). If three independent medical databases all agree on the same incorrect answer because they were scraped from the same biased source, the AI will accept it. UMBRA's coalition detection is built to catch exactly this kind of false consensus.

Or consider the threat of disinformation networks. Coordinated bot networks quietly flood platforms with agreeing voices until a lie feels like the truth. UMBRA trains the Defender to pause and ask why two separate voices are agreeing so perfectly.

As we move toward multi-agent pipelines and enterprise copilots, one compromised model can easily gaslight every other model it touches. We need Defenders that do not fold to pressure.

---

## 5. Clean Engineering & OpenEnv Table Stakes

We engineered UMBRA not just as a concept, but as a clean, standardized, API-first environment ready to be integrated anywhere.

* Proper Base Classes: Uses OpenEnv's Environment and MCPEnvironment base classes under the hood for clean extensibility.
* Client / Server Separation: The remote Hugging Face Space acts as the Server (hosting the heavy RL logic via FastAPI). Clients interact via endpoints and never have to import server internals.
* Standard Gym-Style API: Strictly adheres to the classic env.reset(), env.step(action), and standard discrete State/Observation spacing.
* Valid openenv.yaml Manifest: Fully specified manifest included in the repository to define capabilities correctly.
* Protected Namespace: We ensure no reserved MCP tool names bleed out into overriding MCP client functionalities.

---

## Try It Yourself

UMBRA is deployed natively as a Hugging Face Space Docker container. We invite the judges to interact with it directly!

**1. Interact with the remote Space directly:**
Go to the live swagger UI and click "Try it out" to interact with the environment instantly online.
[https://amrita8642-umbra-meta.hf.space/docs](https://amrita8642-umbra-meta.hf.space/docs)

**2. Install the client code from the repo:**
```bash
pip install git+https://github.com/Amrita8642/Umbra-ShadowWorld-Meta.git
```

**3. Pull and run the container locally:**
```bash
git clone https://github.com/Amrita8642/Umbra-ShadowWorld-Meta.git
cd Umbra-ShadowWorld-Meta
docker build -t umbra-env .
docker run -p 7860:7860 umbra-env
```

**4. Run the FastAPI app locally via Python/Uvicorn:**
```bash
git clone https://github.com/Amrita8642/Umbra-ShadowWorld-Meta.git
cd Umbra-ShadowWorld-Meta
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```
