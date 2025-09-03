## Motivations

Can we build an agent that improves its own performance to reach SOTA on the [ARC-AGI](https://arcprize.org/arc-agi/2/) benchmark?

Consider any of the new coding agents (claude code, gemini cli, openai's codex, etc) -- what is their baseline performance? Can they improve this performance by modifying their own prompts or source code? By how much / to what level of performance does this scale? What is their chosen optimal scaffolding? Can they replicate other methods shared online -- that are known to perform better -- as part of improving their own performance? How long do they take?

Can we produce better "seed" scaffoldings (read: *at least* prompts, though could be more, like models or any glue in-between) that result in more lifetime improvement / scale better or yield better max benchmark performance? If there is a limit, what is it? Why does it occur? And what are the appropriate monitoring infrastructure, scripts, and software to conduct such experiments?

In general, we are interested in [Meta-learning](https://en.wikipedia.org/wiki/Meta-learning_(computer_science)), [Recursive Self-improvement](https://en.wikipedia.org/wiki/Recursive_self-improvement), and [Mesa-optimization](https://www.alignmentforum.org/w/mesa-optimization).
