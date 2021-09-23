# RL_PianoFingering
This is a supplementary material to the submission of the workshop paper (In review):
>Ramoneda, P., Mirón, M., Serra, X. 5th Workshop on Machine Learning for Creativity and Design at NeurIPS 2020 (Sydney, 2021)


## Abstract

Hand and finger movements are a mainstay of piano technique. Automatic Finger-ing from symbolic music data allows us to simulate finger and hand movements.Previous proposals achieve automatic piano fingering based on knowledge-drivenor data-driven techniques. We combine both approaches with deep reinforcementlearning techniques to derive piano fingering. Finally, we explore how to incorpo-rate past experience into reinforcement learning-based piano fingering in furtherwork.

## Experiments

We conduct five experiments to test the behaviour of the fingering algorithm EX1, EX2, EX3, EX4and EX5. EX1 contains a sequence of notes with the same pitch and rhythmic figure. This experimentaims to test whether the RL learning whether the agent learns to use the same finger in every note. InEX2, we have a partial split scale of five ascending notes and the same five descending notes. In thisexperiment, we test whether the RL agent learns not to change the hand position. The EX3 is a pieceof music that does not change the hand throughout its length. Similarly to EX2, EX3 aims at keepingthe same position but in a complex environment. EX is a C major scale. Therefore, the RL agentshould learn to perform only two hand position changes. The fifth test is a piece with the melodicrange of the C major scale. In this case, we want to test whether the RL agent learns to keep the sametwo hand position as EX4 but in a complex environment. All these experiments have been carried outwith various improvements and with different numbers of episodes

Each repo directory contains respectively each experiment.
