import tensorflow as tf
import datetime
from lstm_attention import LSTM_ATT
from lstm import LSTM
import data_helpers
import numpy as np
import time
import os


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", data_helpers.MAX_SENTENCE_LENGTH,
                        "Max sentence length(containing words count) in train/test data")

# Model Hyperparameters
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("text_embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("dist_embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-5, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with.")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")

def train(x_text, y, test_text, test_y):
    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # Create the vocabularyprocessor object, setting the max lengh of the documents.
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    # Transform the documents using the vocabulary.
    text_vec = np.array(list(text_vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))
    print("text_vec = {0}".format(text_vec.shape))

    test_vec = np.array(list(text_vocab_processor.fit_transform(test_text)))
    x_test = test_vec
    y_test = test_y

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    text_shuffled = text_vec[shuffle_indices]
    y_shuffled = np.array(y)[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    text_train, text_dev = text_shuffled[:dev_sample_index], text_shuffled[dev_sample_index:]
    print("text_train: {}!!".format(text_train.shape))
    print("text_dev: {}!!".format(text_dev.shape))

    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    x_train = text_train
    x_dev = text_dev

    print("=====================")
    print("x_train shape = {0}".format(x_train.shape))
    print("y_train shape = {0}".format(y_train.shape))
    print("x_dev shape = {0}".format(x_dev.shape))
    print("y_dev shape = {0}".format(y_dev.shape))
    print("x_test shape = {0}".format(x_test.shape))
    print("y_test shape = {0}".format(np.array(y_test).shape))
    print("=====================")

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print("sequence_length = ", x_train.shape[1])
            # lstm = LSTM_ATT(
            #     sequence_length=x_train.shape[1],
            #     num_classes=y_train.shape[1],
            #     text_vocab_size=len(text_vocab_processor.vocabulary_),
            #     text_embedding_size=FLAGS.text_embedding_dim,
            #     l2_reg_lambda=FLAGS.l2_reg_lambda
            # )
            lstm = LSTM(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                text_vocab_size=len(text_vocab_processor.vocabulary_),
                text_embedding_size=FLAGS.text_embedding_dim,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(lstm.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", lstm.loss)
            acc_summary = tf.summary.scalar("accuracy", lstm.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25,
                                          (len(text_vocab_processor.vocabulary_), FLAGS.text_embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec file {0}".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1')
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = text_vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(lstm.W_text.assign(initW))
                print("Success to load pre-trained word2vec model!\n")

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                # print("x_batch[0]: {}!!".format(x_batch.shape[0]))
                # print("x_batch[1]: {}!!".format(x_batch.shape[1]))
                # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                # print(x_batch)

                feed_dict = {
                    lstm.input_text: x_batch,
                    lstm.input_y: y_batch,
                    lstm.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    lstm.dropout_keep_prob_lstm: 0.3
                }
                # # with attention
                # _, step, summaries, loss, accuracy, vu = sess.run(
                #     [train_op, global_step, train_summary_op, lstm.loss, lstm.accuracy, lstm.vu],
                #     feed_dict)

                # without attention
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, lstm.loss, lstm.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                #                print(len(vu[0]))
                #               print(len(vu))
                #                print(len(output_pos[0]))
                #                print(len(output_pos))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    lstm.input_text: x_batch,
                    # lstm.input_text: x_batch[0],
                    lstm.input_y: y_batch,
                    lstm.dropout_keep_prob: 1.0,
                    lstm.dropout_keep_prob_lstm: 1.0
                }
                # # with attention
                # step, summaries, loss, accuracy, vu = sess.run(
                #     [global_step, dev_summary_op, lstm.loss, lstm.accuracy, lstm.vu],
                #     feed_dict)

                # without attention
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, lstm.loss, lstm.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print(len(vu[0]))
                # print(len(vu))
                # print(vu[0])
                if writer:
                    writer.add_summary(summaries, step)

            def test_step(x_test, y_test, sess):
                """
                Evaluates model on a test set
                """
                feed_dict = {
                    lstm.input_text: x_test,
                    lstm.input_y: y_test,
                    lstm.dropout_keep_prob: 1.0,
                    lstm.dropout_keep_prob_lstm: 1.0
                }

                # # with attention
                # summaries, loss, accuracy, vu = sess.run(
                #     [test_summary_op, lstm.loss, lstm.accuracy, lstm.vu],
                #     feed_dict)

                # without attention
                summaries, loss, accuracy = sess.run(
                    [test_summary_op, lstm.loss, lstm.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("\033[1;33m{}: , loss {:g}, acc {:g}\033[0m".format(time_str, loss, accuracy))
                # print(len(vu[0]))
                # print(len(vu))
                # print(vu[0])
                test_summary_writer.add_summary(summaries)


            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            print("========Train Start!========")
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch = np.array(x_batch)
                # x_batch = np.array(x_batch).transpose()
                # print("222222222222222222222")
                # print(x_batch.shape)

                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\n==========Evaluation==========")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    model_step = current_step
                    latest_model_dir = checkpoint_prefix + "-" + str(model_step)
            print("=======Train DONE!========")
            print("=======Test Start!========")
            saver.restore(sess, latest_model_dir)
            test_step(x_test, y_test, sess)
            

def main(argv=None):
    x_text, y = data_helpers.load_data_and_labels("data/train_text.csv")
    x_test, y_test = data_helpers.load_data_and_labels("data/test_text.csv")
    print("----Data has loaded!----")
    train(x_text, y, x_test, y_test)


if __name__ == "__main__":
    tf.app.run()
