[![Discord](https://img.shields.io/discord/232596713892872193?logo=discord)](https://discord.gg/2JhHVh7CGu)

https://arxiv.org/pdf/2506.11035

A test network for CIFAR10 using Tversky layers. It does indeed learn, seems pretty neat.

To run the Tversky variant train+test.
```
uv run python main.py
```

Tversky Variant Vibe Check:
```
Total Trainable Parameters: 66,147
Overall Accuracy: 64.70%
plane: 64.40%
car: 81.20%
bird: 56.20%
cat: 48.40%
deer: 43.40%
dog: 40.80%
frog: 82.70%
horse: 68.50%
ship: 79.10%
truck: 82.30%
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
