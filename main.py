import framework
import data
import model
import utils


BATCH_SIZE = 32  # batch size: set as large as posible to stabilize training
GEN_STEPS = 1  # generator updates per iteraion: ususally left as 1
CRITIC_STEPS = 10  # critic updates per iteration: use atleast 10 to allow the critic to converge and approximate the Wasserstein distance
GP_WEIGHT = 10  # scaling weight for the gradient penalty loss: 10 is from the original paper
# scaling weight for the cycle consistency loss: 10-100 is usually a good choise
CYCLE_WEIGHT = 10
# generator learning rate: should be pretty small (<1e-4) because momentum based optimizing does not work in a Wasserstein GAN
GEN_LR = 1e-4
CRITIC_LR = 1e-4  # critic learning rate: similar to the generator
ITERATIONS = 300  # number of iterations to train
IMAGE_SIZE = 64


cyclegan = framework.CycleGAN(
    batch_size=BATCH_SIZE,
    gen_steps=GEN_STEPS,
    critic_steps=CRITIC_STEPS,
    gp_weight=GP_WEIGHT,
    cycle_weight=CYCLE_WEIGHT,
)

cyclegan.build(IMAGE_SIZE, gen_lr=GEN_LR, critic_lr=CRITIC_LR)

data_x_train = data.get_dataset(IMAGE_SIZE, shuffle=True, male=True)
data_y_train = data.get_dataset(IMAGE_SIZE, shuffle=True, male=False)

cyclegan.train(data_x_train, data_y_train, ITERATIONS)
cyclegan.save()

data_x_test = data.get_dataset(IMAGE_SIZE, shuffle=False, male=True, test=True)
data_y_test = data.get_dataset(IMAGE_SIZE, shuffle=False, male=False, test=True)

# forward pass: male -> female -> male
fake_y = cyclegan.predict_fake_y(data_x_test)
reco_x = cyclegan.predict_reco_x(data_x_test)
utils.plot_examples(
    data_x_test, fake_y, reco_x, base_name="images/forward/example"
)

# backward pass: female -> male -> female
fake_x = cyclegan.predict_fake_x(data_y_test)
reco_y = cyclegan.predict_reco_y(data_y_test)
utils.plot_examples(
    data_y_test, fake_x, reco_y, base_name="images/backward/example"
)
