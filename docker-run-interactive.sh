wandb docker-run --gpus all --rm --shm-size=20gb -it -v "$(pwd):/app:rw" -w /app worm bash
