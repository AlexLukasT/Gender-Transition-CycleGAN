import framework
import data
import model


BATCH_SIZE = 64  # batch size: set as large as posible to stabilize training
GEN_STEPS = 1  # generator updates per iteraion: ususally left as 1
CRITIC_STEPS = 10  # critic updates per iteration: use atleast 10 to allow the critic to converge and approximate the Wasserstein distance
GP_WEIGHT = 10  # scaling weight for the gradient penalty loss: 10 is from the original paper
# scaling weight for the cycle consistency loss: 10-100 is usually a good choise
CYCLE_WEIGHT = 10
# generator learning rate: should be pretty small (<1e-4) because momentum based optimizing does not work in a Wasserstein GAN
GEN_LR = 1e-4
CRITIC_LR = 1e-4  # critic learning rate: similar to the generator
ITERATIONS = 100  # number of iterations to train
IMAGE_SIZE = 32


cyclegan = framework.CycleGAN(
    batch_size=BATCH_SIZE,
    gen_steps=GEN_STEPS,
    critic_steps=CRITIC_STEPS,
    gp_weight=GP_WEIGHT,
    cycle_weight=CYCLE_WEIGHT,
)

cyclegan.build(IMAGE_SIZE, gen_lr=GEN_LR, critic_lr=CRITIC_LR)

data_x = data.get_dataset(IMAGE_SIZE, shuffle=True, male=True)
data_y = data.get_dataset(IMAGE_SIZE, shuffle=True, male=False)

cyclegan.train(data_x, data_y, ITERATIONS)

