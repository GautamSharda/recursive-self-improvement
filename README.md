## Motivations

Can we build an agent that improves its own performance to reach SOTA on the [ARC-AGI](https://arcprize.org/arc-agi/2/) benchmark? 

Consider any of the new coding agents (claude code, gemini cli, openai's codex, etc) -- what is their baseline performance? Can they improve this performance by modifying their own prompts or source code? By how much? What is the scaffolding they choose? Can they replicate other methods shared online -- that are known to perform better -- to improve their own performance? How efficient (in terms of time and money) are they?

Can we produce better "seed" scaffoldings (read: *at least* prompts, though could be more, like models or any glue in-between) that result in more improvement or better performance? If there is a limit, what is it? Why does it occur? And what are the appropriate monitoring infrastructure, scripts, and software to conduct such experiments?
