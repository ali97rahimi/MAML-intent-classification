from    embed_dataset import EmbedDataset
from    PIL import Image
import  os.path
import  numpy as np


class AmazonNShot:

    def __init__(self, root, filename, batchsz, n_way, k_shot, k_query, embedsz, path='amazon_labse_fa.npy'):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        :para, path:
        """
        

        if not os.path.isfile(os.path.join(root, path)):
            # if root/data.npy does not exist, just create it
            self.x = EmbedDataset(root, filename)

            temp = dict()  # {intent:embed1, embed2..., 100 embeds, intent2: embed1, embed2,... in total, 60 intents for amazon MASSIVE}
            for (embedding, intent) in self.x:
                if intent in temp.keys():
                    temp[intent].append(embedding)
                else:
                    print(intent)
                    temp[intent] = [embedding]

            self.x = []
            for intent, embedding in temp.items():  # intents info deserted , each intent contains 100 embeds
                self.x.append(np.array(embedding))

            print(self.x)

            # as different class may have different number of sentences
            self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
            
            print('data shape:', self.x.shape)  # [150, 100, 768]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, path), self.x)
            print(f'write into {path}.')
        else:
            # if data.npy exists, just load it.
            self.x = np.load(os.path.join(root, path))
            print(f'load from {path}.')

        # For amazon massive dataset
        self.x_train, self.x_test = self.x[:40], self.x[40:]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 150
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        self.embedsz = embedsz  # size of embedding
        # assert (k_shot + k_query) <=100

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),
                               "test": self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):

                    # selected_txt = np.random.choice(100, self.k_shot + self.k_query, False)
                    selected_txt = np.random.choice(28, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_txt[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_txt[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.embedsz)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.embedsz)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.embedsz)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.embedsz)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch


