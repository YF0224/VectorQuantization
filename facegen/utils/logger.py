import os
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, config):
        log_dir = config.get("log", {}).get("log_dir", "runs/default")
        os.makedirs(log_dir, exist_ok=True)
        try:
            self.writer = SummaryWriter(log_dir=log_dir)
        except Exception as e:
            print(f"[Logger] Failed to initialize TensorBoard writer: {e}")
            self.writer = None

    def log_scalar(self, tag, value, step):
        if self.writer:
            try:
                self.writer.add_scalar(tag, value, step)
            except Exception as e:
                print(f"[Logger] Failed to write scalar ({tag}, {value}, {step}): {e}")

    def close(self):
        if self.writer:
            try:
                self.writer.close()
            except Exception as e:
                print(f"[Logger] Failed to close writer: {e}")
