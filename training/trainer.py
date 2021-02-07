import copy
import time

import matplotlib.pyplot as plt
import numpy
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure
from torch import nn

from measures import Measures
from .serialization import ModelSerializer


class Trainer:

    def __init__(self, checkpoint=None, checkpoint_dir=None, best_model_dir=None,
                 model_class=None, model_params=None, best_state_dict=None, loss=None,
                 optimizer_class=None, optimizer_params=None, device=None,
                 params=None, next_epoch=-1, best_epoch=-1, best_dev_accuracy=0,
                 batch_losses=None, avg_train_losses=None,
                 dev_losses=None, avg_dev_losses=None,
                 dev_accuracies=None):
        self.serializer = ModelSerializer(
            checkpoint_dir=checkpoint_dir, best_model_dir=best_model_dir
        )

        # set defaults
        if best_state_dict is None:
            best_state_dict = dict()
        if batch_losses is None:
            batch_losses = []
        if avg_train_losses is None:
            avg_train_losses = []
        if dev_losses is None:
            dev_losses = []
        if avg_dev_losses is None:
            avg_dev_losses = []
        if dev_accuracies is None:
            dev_accuracies = []

        if checkpoint:
            model, optimizer, model_params, optimizer_params, best_state_dict, \
              next_epoch, best_epoch, best_dev_accuracy, \
              batch_losses, avg_train_losses, \
              dev_losses, avg_dev_losses, \
              dev_accuracies, params = self.serializer.load(
                checkpoint, device, model_class, optimizer_class
              )

            self.model = model
            self.model.to(device)
            self.optimizer = optimizer
        else:
            self.model = model_class(model_params)
            self.model.to(device)
            self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

        self.loss = loss
        self.params = params
        self.next_epoch = next_epoch + 1
        self.best_epoch = best_epoch
        self.best_dev_accuracy = best_dev_accuracy
        self.batch_losses = batch_losses
        self.avg_train_losses = avg_train_losses
        self.dev_losses = dev_losses
        self.avg_dev_losses = avg_dev_losses
        self.best_dev_accuracy = best_dev_accuracy
        self.dev_accuracies = dev_accuracies

        self.model_class = model_class
        self.model_params = model_params
        self.best_state_dict = best_state_dict

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        self.register_hooks()
        self.model_summary()

    def model_summary(self):
        print(self.model)
        print()

        number_of_trainable_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Number of trainable parameters {0:,}".format(
            number_of_trainable_parameters
        ))
        print()

        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(" ", name)
        print()

    def early_stop(self, patience):
        stop = False
        patience += 1
        if len(self.avg_dev_losses) > patience and \
                min(self.avg_dev_losses[-patience:]) == self.avg_dev_losses[-patience]:
            stop = True
        return stop

    def register_hooks(self):
        def is_bad_grad(grad):
            grad = grad.data
            if grad.ne(grad).any() or grad.gt(1e6).any():
                print("WARNING: Bad gradient ", grad)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(is_bad_grad)

    def plot_training_losses(self, labels=None, path=None):
        figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='w')

        plt.plot(range(1, len(self.avg_train_losses) + 1), self.avg_train_losses, label=labels['avg_train_losses'])
        plt.plot(range(1, len(self.avg_dev_losses) + 1), self.avg_dev_losses, label=labels['avg_dev_losses'])
        plt.plot(range(1, len(self.dev_accuracies) + 1), self.dev_accuracies, label=labels['dev_accuracies'])

        plt.xlabel(labels['xlabel'])

        # find position of lowest dev loss
        min_avg_dev_loss = self.avg_dev_losses.index(min(self.avg_dev_losses)) + 1
        plt.axvline(min_avg_dev_loss, linestyle='--', color='r', label=labels['min_avg_dev_loss'])

        # find position of max dev accuracy
        max_dev_accuracy = self.dev_accuracies.index(max(self.dev_accuracies)) + 1
        plt.axvline(max_dev_accuracy, linestyle='--', color='c', label=labels['max_dev_accuracy'])

        plt.grid(True)
        plt.legend()

        plt.savefig(path)
        plt.show()

    def plot_grad_flow(self):
        """Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""

        average_grads, max_grads = [], []
        layers = []

        for name, param in self.model.named_parameters():
            if param.requires_grad and "bias" not in name:
                layers.append(name)
                average_grads.append(param.grad.abs().mean())
                max_grads.append(param.grad.abs().max())

        plt.bar(numpy.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(numpy.arange(len(max_grads)), average_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(average_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(average_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(average_grads))
        plt.ylim(bottom=-0.0001, top=0.5)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend(
            [
                Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)
            ],
            ['max-gradient', 'mean-gradient', 'zero-gradient'],
            loc='upper right'
        )

    def fit(self, train_dataloader=None, dev_dataloader=None):

        # first_batch = next(iter(train_dataloader))
        # print (first_batch)

        start = time.time()
        for epoch in range(self.next_epoch, self.params['num_epochs']):
            t0 = time.time()

            self.model.train()  # set the model to training mode
            # for data in [first_batch] * 1:
            for data in train_dataloader:
                self.model.zero_grad()
                output = self.model(data['x'], data['length'])
                batch_loss = self.loss(output, data['y'])
                self.batch_losses.append(batch_loss.item())
                batch_loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.params['max_norm'])
                # self.plot_grad_flow()
                self.optimizer.step()

            self.avg_train_losses.append(numpy.average(self.batch_losses))

            self.model.eval()

            predictions, labels = [], []
            for data in dev_dataloader:
                output = self.model(data['x'], data['length'])
                dev_batch_loss = self.loss(output, data['y'])
                self.dev_losses.append(dev_batch_loss.item())
                argmax = output.argmax(dim=1).detach().cpu().numpy()
                predictions.extend(list(argmax))
                labels.extend(list(data['y'].detach().cpu().numpy()))

            _, _, _, _, _, _, _, dev_accuracy = Measures.accuracy(predictions, labels)
            self.dev_accuracies.append(dev_accuracy)

            dev_loss = numpy.average(self.dev_losses)
            self.avg_dev_losses.append(dev_loss)

            if dev_accuracy > self.best_dev_accuracy:
                self.best_dev_accuracy = dev_accuracy
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())

            print(
                "Epoch {}/{} : dev accuracy: {:.2f}\tdev loss: {:.2f}\t"
                "best dev accuracy: {:.2f}\tbest epoch: {}\tTime: {:.2f}s".format(
                    epoch, self.params['num_epochs'], dev_accuracy, dev_loss,
                    self.best_dev_accuracy, self.best_epoch, time.time() - t0
                )
            )

            self.serializer.save(
                model=self.model, model_params=self.model_params,
                best_state_dict=self.best_state_dict,
                optimizer=self.optimizer,
                optimizer_params=self.optimizer_params,
                epoch=epoch, best_epoch=self.best_epoch,
                best_dev_accuracy=self.best_dev_accuracy,
                batch_losses=self.batch_losses,
                avg_train_losses=self.avg_train_losses,
                dev_losses=self.dev_losses,
                avg_dev_losses=self.avg_dev_losses,
                dev_accuracies=self.dev_accuracies,
                params=self.params
            )

            if self.early_stop(patience=self.params['patience']):
                print('Early stopping...')
                break

        self.model.load_state_dict(self.best_state_dict)
        print("Best epoch: {}\tBest dev accuracy: {:.2f}\tTime: {:.2f}s".format(
            self.best_epoch, self.best_dev_accuracy, time.time() - start
        ))

        return self.model
