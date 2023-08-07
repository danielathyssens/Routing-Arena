from torch import nn


class TemplateModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 normalization='batch'):
        super(TemplateModel, self).__init__()

        raise NotImplementedError


def train_epoch(model, optimizer, lr_scheduler, epoch, train_dataset, val_dataset,
                tb_logger, ckpt_save_path, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    raise NotImplementedError
