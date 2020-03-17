from mnist import MNIST
from inference import Inference
from config import Config

execute = {
    'training and activations data generation': True,
    'inference': True,
}

config = Config()

if execute['training and activations data generation']:
    mnist = MNIST(config)
    mnist.run_training()
    mnist.generate_activations_data()

if execute['inference']:
    infer = Inference(config.use_model)
    infer.get_performance_details()
