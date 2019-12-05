import tensorflow as tf
import model


class CycleGAN:
    def __init__(
        self,
        batch_size=64,
        gen_steps=1,
        critic_steps=10,
        gp_weight=10,
        cycle_weight=10,
    ):
        self.batch_size = batch_size
        self.gen_steps = gen_steps
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight
        self.cycle_weight = cycle_weight

        # set in sefl.build()
        self.writer = None
        self.generator_g = None
        self.generator_f = None
        self.critic_x = None
        self.critic_y = None
        self.gen_g_optimizer = None
        self.gen_f_optimizer = None
        self.critic_x_optimizer = None
        self.critic_y_optimizer = None

    def build(self, image_size, gen_lr=1e-4, critic_lr=1e-4):
        # summary writer for tensorboard
        self.writer = tf.summary.create_file_writer("summary")

        # generator keras models
        self.generator_g = model.build_generator(image_size)
        self.generator_f = model.build_generator(image_size)

        # critic keras models
        self.critic_x = model.build_critic(image_size)
        self.critic_y = model.build_critic(image_size)

        # optimizers for generators and critic
        # Due to the Generator and Critic loss influencing each other the loss phase space
        # is not a scalar field anymore but a vector field.
        # Therefore momentum based optimizing does not really work anymore and the
        # momentum parameters beta_1 and beta_2 are set pretty low,
        # usually: beta_1=0.9, beta_2=0.999
        # The learning rates should also be chosen relatively small (<1e-4) because of this
        self.gen_g_optimizer = tf.optimizers.Adam(
            gen_lr, beta_1=0.5, beta_2=0.9
        )
        self.gen_f_optimizer = tf.optimizers.Adam(
            gen_lr, beta_1=0.5, beta_2=0.9
        )
        self.critic_x_optimizer = tf.optimizers.Adam(
            critic_lr, beta_1=0.5, beta_2=0.9
        )
        self.critic_y_optimizer = tf.optimizers.Adam(
            critic_lr, beta_1=0.5, beta_2=0.9
        )

    def summary(self, tag, value, step):
        # write a scalar value (loss) to the summary file
        true_step = tf.cast(step / self.critic_steps, tf.int64)
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=true_step)

    def generator_loss(self, critic_out_fake, direction, step):
        # generator loss: minimize -D(G(z))
        loss = -tf.reduce_mean(critic_out_fake)
        tag = direction + "/gen_loss"
        if step % self.critic_steps < 1:
            self.summary(tag, loss, step)
        return loss

    def critic_loss(self, critic_out_real, critic_out_fake, direction, step):
        # critic loss: maximize D(x) - D(G(z)) <=> minimize D(G(z)) - D(x)
        loss = tf.reduce_mean(critic_out_fake) - tf.reduce_mean(critic_out_real)
        tag = direction + "/critic_loss"
        if step % self.critic_steps < 1:
            self.summary(tag, loss, step)
        return loss

    def gradient_penalty_loss(self, fake, real, direction, step):
        # gradient penalty loss to enfore 1-Lipschitz constraint in the critic:
        # take random points on a virtual connection between the generated and the real sample
        # and enfore the critic gradient to be 1 on this point
        critic = self.critic_x if direction == "x" else self.critic_y
        tag = (
            "y2x/gradient_penalty"
            if direction == "x"
            else "x2y/gradient_penalty"
        )

        epsilon = tf.random.uniform(
            [self.batch_size, 1, 1, 1, 1], minval=0, maxval=1
        )
        averaged_batch = epsilon * real + (1 - epsilon) * fake
        averaged_batch_out = critic(averaged_batch, training=True)
        gradients = tf.gradients(averaged_batch_out, averaged_batch)[0]
        normed_gradients = tf.sqrt(
            tf.reduce_sum(tf.square(gradients), axis=[1, 2])
        )
        loss = self.gp_weight * tf.reduce_mean((normed_gradients - 1) ** 2)

        if step % self.critic_steps < 1:
            self.summary(tag, loss, step)
        return loss

    def cycle_consistency_loss(self, real, reconstructed, direction, step):
        # cycle consistency loss: punish reconstruced images that differ from the real ones
        loss = self.cycle_weight * tf.reduce_mean(tf.abs(real - reconstructed))
        tag = "cycle_consistency_loss/" + direction
        if step % self.critic_steps < 1:
            self.summary(tag, loss, step)
        return loss

    @tf.function
    def train_step(self, real_x, real_y, step):
        # run one update step of the network by passing one batch through the graph
        # to calculate the gradients and applying them
        with tf.GradientTape(persistent=True) as tape:
            # forward pass
            fake_y = self.generator_g(real_x, training=True)
            reconstructed_x = self.generator_f(fake_y, training=True)
            real_y_out = self.critic_y(real_y, training=True)
            fake_y_out = self.critic_y(fake_y, training=True)

            gen_loss_g = self.generator_loss(fake_y_out, "x2y", step)
            cycle_loss_x = self.cycle_consistency_loss(
                real_x, reconstructed_x, "x2x", step
            )
            critic_y_loss = self.critic_loss(
                real_y_out, fake_y_out, "x2y", step
            )
            critic_y_loss += self.gradient_penalty_loss(
                fake_y, real_y, "y", step
            )

            # backward pass
            fake_x = self.generator_f(real_y, training=True)
            reconstructed_y = self.generator_g(fake_x, training=True)
            real_x_out = self.critic_x(real_y, training=True)
            fake_x_out = self.critic_x(fake_x, training=True)

            gen_loss_f = self.generator_loss(fake_x_out, "y2x", step)
            cycle_loss_y = self.cycle_consistency_loss(
                real_y, reconstructed_y, "y2y", step
            )
            critic_x_loss = self.critic_loss(
                real_x_out, fake_x_out, "y2x", step
            )
            critic_x_loss += self.gradient_penalty_loss(
                fake_x, real_x, "x", step
            )

            # add total cycle consistency loss to both generator losses
            total_cycle_loss = cycle_loss_x + cycle_loss_y
            total_gen_g_loss = gen_loss_g + total_cycle_loss
            total_gen_f_loss = gen_loss_f + total_cycle_loss

        # train both generators every X critic steps
        if step % self.critic_steps < 1:
            gen_g_gradients = tape.gradient(
                total_gen_g_loss, self.generator_g.trainable_variables
            )
            gen_f_gradients = tape.gradient(
                total_gen_f_loss, self.generator_f.trainable_variables
            )
            self.gen_g_optimizer.apply_gradients(
                zip(gen_g_gradients, self.generator_g.trainable_variables)
            )
            self.gen_f_optimizer.apply_gradients(
                zip(gen_f_gradients, self.generator_f.trainable_variables)
            )

        # train both critics
        critic_x_gradients = tape.gradient(
            critic_x_loss, self.critic_x.trainable_variables
        )
        critic_y_gradients = tape.gradient(
            critic_y_loss, self.critic_y.trainable_variables
        )

        self.critic_x_optimizer.apply_gradients(
            zip(critic_x_gradients, self.critic_x.trainable_variables)
        )

        self.critic_y_optimizer.apply_gradients(
            zip(critic_y_gradients, self.critic_y.trainable_variables)
        )

    def train(self, dataset_x, dataset_y, iterations):
        real_iterations = iterations * self.critic_steps
        zip_dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
        train_dataset = (
            zip_dataset.batch(self.batch_size).prefetch(1).take(real_iterations)
        )
        for i, (data_x, data_y) in enumerate(train_dataset):
            step = tf.constant(i, dtype=tf.int64)
            self.train_step(data_x, data_y, step)
