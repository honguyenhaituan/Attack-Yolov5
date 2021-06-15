from pathlib import Path
from utils.general import colorstr


try:
    import wandb
    from wandb import init, finish
except ImportError:
    wandb = None

class WandbLogger():
    def __init__(self, opt, name, run_id, epoch): 
        self.wandb, self.wandb_run = wandb, None if not wandb else wandb.run
        if self.wandb: 
            self.wandb_run = wandb.init(config=opt,
                                        project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                                        name=name,
                                        id=run_id) if not wandb.run else wandb.run
        self.log_imgs = 100
        self.current_epoch = 0
        self.epoch_dict = [{} for _ in range(epoch)]

        if not self.wandb_run:
            prefix = colorstr('wandb: ')
            print(f"{prefix}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)")

    def log(self, log_dict):
        self.log_epoch(self.current_epoch, log_dict)

    def increase_log(self, log_dict):
        self.increase_log_epoch(self.current_epoch, log_dict)

    def end_epoch(self): 
        self.current_epoch += 1

    def log_epoch(self, epoch, log_dict):
        if self.wandb_run:
            for key, value in log_dict.items():
                self.epoch_dict[epoch][key] = value

    def increase_log_epoch(self, epoch, log_dict): 
        if self.wandb_run:
            for key, value in log_dict.items(): 
                self.epoch_dict[epoch][key] = self.epoch_dict[epoch][key] + value if key in self.epoch_dict[epoch] else value

    def finish_run(self):
        if self.wandb_run:
            if self.epoch_dict:
                for _log in self.epoch_dict:
                    wandb.log(_log)
            wandb.run.finish()