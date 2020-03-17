import torch
from model_utils import test
from models import ModelFactory
from mnist import MNIST
from config import Config


class Inference:

    def __init__(self, model_name):
        self.model_name = model_name

    def get_performance_details(self):
        config = Config()
        infer_model = ModelFactory(self.model_name, config.layers).get_model()
        mnist = MNIST(config)
        infer_model.load_state_dict(torch.load(mnist.config.ROOT_DIR + '/saved_models/mnist_' +
                                               self.model_name + '_' + str(config.n_acts) + str(config.model_id)
                                               + '.pt', map_location=mnist.config.device))

        test_loader = mnist.get_test_loader()
        accuracy, missclassified = test(mnist.config, infer_model, test_loader)

        return accuracy, missclassified
