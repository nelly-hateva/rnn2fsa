import os

import torch


class ModelSerializer:

    def __init__(self, checkpoint_dir=None, best_model_dir=None):
        self.checkpoint_dir = checkpoint_dir
        self.best_model_dir = best_model_dir
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            if not os.path.exists(self.checkpoint_dir):
                raise "Directory {} doesn't exist".format(self.checkpoint_dir)
        if self.best_model_dir:
            os.makedirs(self.best_model_dir, exist_ok=True)
            if not os.path.exists(self.best_model_dir):
                raise "Directory {} doesn't exist".format(self.best_model_dir)

    @staticmethod
    def load(path, device, model_class, optimizer_class):

        state = torch.load(path, map_location=device)

        model_params = state['model_params']
        model = model_class(model_params)
        model.load_state_dict(state['state_dict'])
        model.to(device)

        optimizer_params = state['optimizer_params']
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        optimizer.load_state_dict(state['optimizer_state_dict'])

        return model, optimizer, model_params, optimizer_params, \
            state['best_state_dict'], state['epoch'], \
            state['best_epoch'], state['best_dev_accuracy'], \
            state['batch_losses'], state['avg_train_losses'], \
            state['dev_losses'], state['avg_dev_losses'], \
            state['dev_accuracies'], state['params']

    @staticmethod
    def load_model(path, device, model_class):

        state = torch.load(path, map_location=device)

        model_params = state['model_params']
        model = model_class(model_params)
        model.load_state_dict(state['state_dict'])
        model.to(device)

        return model

    def save(self, model=None, model_params=None, best_state_dict=None,
             optimizer=None, optimizer_params=None,
             epoch=None, best_epoch=None, best_dev_accuracy=None,
             batch_losses=None, avg_train_losses=None,
             dev_losses=None, avg_dev_losses=None,
             dev_accuracies=None, params=None
             ):
        state = {
            'model_params': model_params,
            'state_dict': model.state_dict(),
            'best_state_dict': best_state_dict,
            'optimizer_params': optimizer_params,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_dev_accuracy': best_dev_accuracy,
            'batch_losses': batch_losses,
            'avg_train_losses': avg_train_losses,
            'dev_losses': dev_losses,
            'avg_dev_losses': avg_dev_losses,
            'dev_accuracies': dev_accuracies,
            'params': params
        }

        if self.checkpoint_dir:
            torch.save(state, os.path.join(self.checkpoint_dir, "checkpoint.pt"))
        if self.best_model_dir and epoch == best_epoch:
            torch.save(state, os.path.join(self.best_model_dir, "model.pt"))
