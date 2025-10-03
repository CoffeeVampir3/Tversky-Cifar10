[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

https://arxiv.org/pdf/2506.11035

XOR test: https://github.com/CoffeeVampir3/Tverysky-Torch
Cifar10: https://github.com/CoffeeVampir3/Tversky-Cifar10/tree/main
Language model: https://github.com/CoffeeVampir3/Architecture-Tversky-All

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
