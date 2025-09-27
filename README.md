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

Not present in the paper, an experimental multihead tversky which partitions prototypes. So 2 heads with 20 prototypes is 10 distinct prototypes per head.

This particular test configuration is: `[Hidden=64, heads=2, prototypes=24, features=12]`
```
uv run python multihead.py
```
```
Total Trainable Parameters: 51,430
Training finished in 66.56s
Overall Accuracy: 64.04%
plane: 54.60%
car: 74.60%
bird: 32.20%
cat: 58.80%
deer: 57.50%
dog: 56.30%
frog: 77.70%
horse: 59.00%
ship: 86.20%
truck: 83.50%

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
