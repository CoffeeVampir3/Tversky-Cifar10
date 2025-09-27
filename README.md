[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

https://arxiv.org/pdf/2506.11035

A test network for CIFAR10 using Tversky layers. It does indeed learn, seems pretty neat.

To run the Tversky variant train+test.
```
uv run python main.py
```

Tversky Variant Vibe Check:
```
Total Trainable Parameters: 50,403
Overall Accuracy: 64.70%
plane: 74.40%
car: 80.60%
bird: 48.20%
cat: 42.80%
deer: 53.10%
dog: 61.30%
frog: 74.90%
horse: 61.40%
ship: 68.20%
truck: 82.10%
```

I'm experimenting with a multihead version that works similarly to multiheaded attention, we can distribute the hidden dimension over multiple heads to make things more computationally efficient. E.g.: `self.tversky = TverskyMultihead(64, 2, 10, 10)` is a 64 hidden dim with 2 heads, so we end up with each head accounting for a hidden dim of 32 here with 10 prototypes and 10 features. At this network size it's objectively worse to distribute the hidden like this, but can be used in larger networks to reduce noise similar to MHA. Additionally, we can softmax here and get something along the lines of a Tversky Multihead Attention.

At a head size of 1; `self.tversky = TverskyMultihead(64, 1, 10, 10)` -- this is equivalent to a regular TverskyLayer. So if your Tversky is getting large enough to be intractable you can distribute over heads to make it possible to calculate without it being insanely slow or allocate massive memory.
```
uv run python multihead.py
```

```
Total Trainable Parameters: 50,403
Overall Accuracy: 55.76%
plane: 58.70%
car: 81.60%
bird: 34.00%
cat: 41.50%
deer: 50.70%
dog: 33.80%
frog: 83.40%
horse: 47.60%
ship: 70.80%
truck: 55.50%
```

To run the control network train+test.
```
uv run python control.py
```

Control Network Vibe Check:
```
Total Trainable Parameters: 49,770
Overall Accuracy: 52.08%
plane: 56.50%
car: 58.40%
bird: 41.30%
cat: 39.80%
deer: 48.80%
dog: 26.70%
frog: 77.80%
horse: 41.20%
ship: 73.90%
truck: 56.40%
```
