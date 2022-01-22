import numpy as np
import subprocess
import tensorflow as tf

from constraints import TARGET
from keras.models import load_model
from visualization import plot_binary_mask, example_create


class Surrogate:

    def __init__(self,
                 surrogate,
                 assistant,
                 prepared_weight=False):
        self.state = False
        self.classification_threshold = 0
        self.surrogate = surrogate
        self.assistant = assistant
        self.prepared_weight = prepared_weight
        self.real = RealModel()

    def __call__(self, population):
        hs_pop = []
        counter_swan = 0

        for individ in population:
            binary_mask = plot_binary_mask(individ)
            pred_for_individ = self.assistant.predict(binary_mask.reshape(1, 224, 224))

            if pred_for_individ[0][0] > self.classification_threshold:
                counter_swan += 1
                _, hs_for_ind, _ = self.real([individ])
                hs_pop.append(hs_for_ind[0])

            else:
                _, hs_for_surr, _ = self.surrogate_modeling_ind(binary_mask)
                hs_pop.append(round(hs_for_surr[0], 6))

        return None, hs_pop, counter_swan

    def dataset_preparation(self, pop, Z_pop, hs_pop):
        def train_test_split(ex_idx):
            train_part = int(len(ex_idx) * 0.8)
            np.random.shuffle(ex_idx)

            train_idx = ex_idx[:train_part]
            test_idx = ex_idx[train_part:]

            return np.array(train_idx), np.array(test_idx)

        _ = [example_create(Z_pop[i], pop[i], i) for i, ind in enumerate(pop)]

        def data_augm(image_name, label_hs):
            label = label_hs[0]
            hs = tf.cast(label_hs[1], dtype=tf.float32)

            direct_feat = path_to_dataset + 'targets/'
            direct_labels = path_to_dataset + 'labels/'

            idx_feat = image_name
            image_feat = tf.io.read_file(direct_feat + idx_feat + '.png')
            image_feat = tf.image.decode_png(image_feat, channels=1)
            image_feat = tf.image.resize(image_feat, (224, 224))

            idx_label = label
            image_label = tf.io.read_file(direct_labels + idx_label + '.png')
            image_label = tf.image.decode_png(image_label, channels=1)
            image_label = tf.image.resize(image_label, (224, 224))

            image_label_hs = (image_label, hs)

            return (image_feat, image_label_hs)

        path_to_dataset = 'dataset/'
        dataset_size = len(pop)
        hs = hs_pop
        examples_idx = list(range(dataset_size))

        train_idx, test_idx = train_test_split(examples_idx)
        train_names = train_idx.astype(str)
        test_names = test_idx.astype(str)

        hs_train = hs[train_idx]
        hs_val = hs[test_idx]

        train_ds_idx = tf.data.Dataset.from_tensor_slices(list(train_names))
        train_ds_labels = tf.data.Dataset.from_tensor_slices((list(train_names), hs_train))
        train_ds = tf.data.Dataset.zip((train_ds_idx, train_ds_labels))

        val_ds_idx = tf.data.Dataset.from_tensor_slices(list(test_names))
        val_ds_labels = tf.data.Dataset.from_tensor_slices((list(test_names), hs_val))
        val_ds = tf.data.Dataset.zip((val_ds_idx, val_ds_labels))

        train_ds = train_ds.map(data_augm)
        val_ds = val_ds.map(data_augm)

        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), (normalization_layer(y[0]), y[1])))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), (normalization_layer(y[0]), y[1])))

        return train_ds, val_ds

    def assistant_data_preparation(self, ds, thresh):
        new_ds = []

        for example in ds:
            feature = example[0].numpy().reshape(224, 224, 1)
            hs_val = example[1][1]
            hs_pred = self.surrogate.predict(feature.reshape(1, 224, 224, 1))[1]

            mae = abs(hs_val - hs_pred)
            if mae[0].numpy() >= thresh:
                label = 1
            elif mae[0].numpy() < thresh:
                label = 0

            new_ds.append((feature, label))

        return new_ds

    def preparation(self, pop, Z_pop, hs_pop, pretrained=False):
        if pretrained:
            self.surrogate = load_model(self.prepared_weight[0])
            self.assistant = load_model(self.prepared_weight[1])
            self.classification_threshold = 0.23
            self.state = True
        else:
            train_ds, val_ds = self.dataset_preparation(pop, Z_pop, hs_pop)

            opt = tf.keras.optimizers.Adam()
            self.surrogate.compile(optimizer=opt, loss={'decoded': 'binary_crossentropy', 'linear': 'mae'},
                                   metrics={'decoded': 'mse', 'linear': 'mae'})

            es = tf.keras.callbacks.EarlyStopping(
                monitor='val_linear_mae', patience=4, verbose=0,
                mode='min', restore_best_weights=True)

            plateau = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_linear_mae', factor=0.95, patience=5, verbose=0,
                mode='min')

            self.surrogate.fit(train_ds.batch(12),
                               epochs=35,
                               batch_size=12,
                               callbacks=[es, plateau],
                               validation_data=val_ds.batch(12))

            ass_train_ds = self.assistant_data_preparation(train_ds, 0.05)
            ass_val_ds = self.assistant_data_preparation(val_ds, 0.05)

            X = [ex[0] for ex in ass_train_ds]
            X = np.array(X).reshape(-1, 224, 224, 1)
            Y = np.array([ex[1] for ex in ass_train_ds]).reshape(-1, 1)

            X_val = np.array([ex[0] for ex in ass_val_ds])
            Y_val = np.array([ex[1] for ex in ass_val_ds]).reshape(-1, 1)
            val_ds_new = (X_val, Y_val)

            self.assistant.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(),
                                   metrics=[tf.keras.metrics.AUC(from_logits=True)])

            es = tf.keras.callbacks.EarlyStopping(
                monitor='val_auc_4', patience=10, verbose=0,
                mode='max', restore_best_weights=True)

            self.assistant.fit(x=X,
                               y=Y,
                               epochs=40,
                               batch_size=16,
                               callbacks=[es],
                               validation_data=val_ds_new)

            self.state = True

    def surrogate_modeling_ind(self, binary_mask):
        hs_target = []

        map_for_individ = self.surrogate.predict(binary_mask.reshape(1, 224, 224))
        hs_target.append(map_for_individ[1][0])

        return None, hs_target, 0


class RealModel:

    def __init__(self):
        self.path_to_model = 'swan/'
        self.path_to_input = 'swan/INPUT'
        self.path_to_hs = 'swan/r/hs47dd8b1c0d4447478fec6f956c7e32d9.d'

    def __call__(self, population):
        hs_target = []
        Z = []

        for individ in population:
            file_to_read = open(self.path_to_input, 'r')
            content_read = file_to_read.read()

            for_input = '\nOBSTACLE TRANSM 0. REFL 0. LINE '
            num_of_bw = len(individ)
            for j, ind in enumerate(individ):
                num_of_points = len(ind)
                for i, gen in enumerate(ind):
                    if (i + 1) % 2 == 0:
                        if (i + 1) == num_of_points:
                            for_input += str(1450 - gen)
                        else:
                            for_input += str(1450 - gen) + ', '
                    else:
                        for_input += str(gen) + ', '

                if j == (num_of_bw - 1):
                    for_input += '\n$optline'
                else:
                    for_input += '\nOBSTACLE TRANSM 0. REFL 0. LINE '

            content_to_replace = for_input
            content_write = content_read.replace(
                content_read[content_read.find('\n\n\n') + 3:content_read.rfind('\n$optline') + 9], content_to_replace)
            file_to_read.close()

            file_to_write = open(self.path_to_input, 'w')
            file_to_write.writelines(content_write)
            file_to_write.close()

            subprocess.call('swan.exe', shell=True, cwd=self.path_to_model)

            hs = np.loadtxt(self.path_to_hs)

            Z_new = []
            for z in hs:
                z_new = []
                for k in z:
                    if k <= 0:
                        z_new.append(0)
                    else:
                        z_new.append(k)
                Z_new.append(z_new)
            Z_new = np.array(Z_new)

            Z.append(Z_new)
            hs_target.append((hs[TARGET[0][0], TARGET[0][1]] + hs[TARGET[1][0], TARGET[1][1]]) / 2)

        return Z, hs_target, len(population)
