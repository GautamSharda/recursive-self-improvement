## Motivation

One of the key drivers of AI progress has been researchers attempting to achieve SOTA on benchmarks. How good is AI at doing this? In particular, can we build an agent that improves its own performance to reach SOTA on the [ARC-AGI](https://arcprize.org/arc-agi/2/) benchmark?

Consider any of the new coding agents (claude code, gemini cli, openai's codex, etc) -- what is their baseline performance? Can they improve this performance by modifying their own prompts or source code? By how much / to what level of performance does this scale? What is their chosen optimal scaffolding? Can they replicate other methods shared online -- that are known to perform better -- as part of improving their own performance? How long do they take?

Can we produce better "seed" scaffoldings (read: *at least* prompts, though could be more, like models or any glue in-between) that result in more lifetime improvement / scale better or yield better max benchmark performance? If there is a limit, what is it? Why does it occur? And what are the appropriate monitoring infrastructure, scripts, and software to conduct such experiments?

In general, we are interested in [Meta-learning](https://en.wikipedia.org/wiki/Meta-learning_(computer_science)), [Reinforcement-learning](https://en.wikipedia.org/wiki/Reinforcement_learning), [Online-learning](https://en.wikipedia.org/wiki/Online_machine_learning), [Large Language Models](https://en.wikipedia.org/wiki/Large_language_model), [Recursive Self-Improvement](https://en.wikipedia.org/wiki/Recursive_self-improvement), and [Mesa-Optimization](https://www.alignmentforum.org/w/mesa-optimization).

## Roadmap

**Note: This is always subject to change. I used Claude to make one for now.**

Day 1: Baseline & Environment Questions

What is the current baseline performance of [chosen coding agent] on ARC-AGI tasks?
How many problems can it solve correctly out of a sample of 20 tasks?
What is the average time per problem attempt?
What prompting strategy does it currently use by default?

Day 2: Self-Modification Capability Questions

Can the agent analyze its own failures and identify why it got problems wrong?
Can it successfully modify its own system prompt based on failure analysis?
Does performance improve, stay the same, or degrade after self-modification?
How many iteration cycles does it take to see meaningful change?

Day 3: Method Replication Questions

Can the agent understand and implement existing ARC-AGI solution methods when given their descriptions?
Which published methods can it successfully replicate vs. which ones does it struggle with?
Do the replicated methods perform better than its original approach?

Day 4: Scaffolding Comparison Questions

Which initial scaffolding approach (prompt style, reasoning framework, tool access) leads to better baseline performance?
Which scaffolding enables the most improvement through self-modification?
Is there a clear "winner" or do different scaffoldings excel at different problem types?

Day 5: Scaling & Limits Questions

At what point does performance plateau in the self-improvement loop?
What causes the plateau - is it a fundamental limit or a bug in the approach?
How does performance scale when moving from 20 to 50+ problems?
What failure modes emerge during extended self-improvement chains?
