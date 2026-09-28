# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
from mmcv.runner import Hook, HOOKS, master_only, LoggerHook


@HOOKS.register_module()
class CustomWandbLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 commit=True,
                 by_epoch=True,
                 with_step=True):
        super(CustomWandbLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.import_wandb()
        self.init_kwargs = init_kwargs
        self.commit = commit
        self.with_step = with_step

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    @master_only
    def before_run(self, runner):
        super(CustomWandbLoggerHook, self).before_run(runner)
        if self.wandb is None:
            self.import_wandb()
        init_kwargs = {} if self.init_kwargs is None else self.init_kwargs
        init_kwargs['id'] = runner.meta["wandb_id"] if "wandb_id" in runner.meta else None  # Id for resume
        init_kwargs['resume'] = "allow"
        self.wandb.init(**init_kwargs)
        runner.meta['wandb_id'] = self.wandb.run.id

    @master_only
    def log(self, runner, epoch, commit=True):
        tags = self.get_loggable_tags(runner, epoch=epoch)
        if tags:
            if self.with_step:
                self.wandb.log(
                    tags, step=self.get_iter(runner), commit=commit)
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=commit)

    @master_only
    def after_run(self, runner):
        self.wandb.join()

    def get_loggable_tags(self,
                          runner,
                          allow_scalar=True,
                          allow_text=False,
                          add_mode=True,
                          tags_to_skip=('time', 'data_time'),
                          epoch=False):
        if epoch:
            postfix = 'epoch'
        else:
            postfix = 'iter'
        tags = {}
        for var, val in runner.log_buffer.output.items():
            if var in tags_to_skip:
                continue
            if self.is_scalar(val) and not allow_scalar:
                continue
            if isinstance(val, str) and not allow_text:
                continue
            if add_mode:
                var = f'{self.get_mode(runner)}{var}/{postfix}'
                var = var.replace('trainmetrics', 'val')  # hacky fix to report the val metrics under val and not train
            tags[var] = val
        tags.update(self.get_lr_tags(runner))
        tags.update(self.get_momentum_tags(runner))
        tags['epoch'] = self.get_epoch(runner)
        return tags

    def after_train_iter(self, runner):
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner, epoch=False)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        runner.log_buffer.average()
        self.log(runner, epoch=True, commit=False)
        if self.reset_flag:
            runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        runner.log_buffer.average()
        self.log(runner, epoch=True)
        if self.reset_flag:
            runner.log_buffer.clear_output()
