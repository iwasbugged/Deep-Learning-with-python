class TimeSeriesGenerator:
    """
    It yields a tuple (samples , targets), where 'samples' is one batch of input data and targets is the 
    corrosponding array of targets temperatures. It takes the following arguments
    """

    def __init__(self , data, lookback , delay , min_index , max_index , shuffle = False , batch_size = 128 , step = 6):
        '''
        Data:     The original array of data from which data is to be generated.
        Lookback: How many timesteps back the input data should go.
        Delay:    How many timesteps in the future the target should be.
        Min_index and Max_index: Indices in the data array that delimit which timesteps to draw from. This is useful for 
                                 keeping a segment of the data for validation and another for testing.
        Shuffle:  Whether to shuffle the samples or draw them in chronology order.
        Batch_size: The number of samples per batch.
        Steps: The period, in timesteps at which we sample data. We'll set it to 6 in order to draw one data point every 
        hour
        '''
        self.data = data
        self.lookback = lookback
        self.delay = delay
        self.min_index = min_index
        self.max_index = max_index
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.step = step
    @staticmethod
    def Generator(self):
        import numpy as np

        if self.max_index is None:
            self.max_index = len(self.data) - self.delay -1

        i = self.min_index + self.lookback

        while 1 :
            if self.shuffle:
                rows = np.random.randint(self.min_index + self.lookback , self.max_index , size = self.batch_size)

            else :
                if i + self.batch_size >= self.max_index:
                    i = self.min_index + self.lookback

                rows = np.arange( i , min(i + self.batch_size , self.max_index))

                i += len(rows)

            samples = np.zeros((len(rows) , self.lookback // self.step , self.data.shape[-1]))
            targets = np.zeros((len(rows), ))

            for j , rows in enumerate(rows):
                indices = range(rows[j] - self.lookback , rows[j] , self.step)

                samples[j] = self.data[indices]
                targets[j] = self.data[rows[j] + self.delay][1]

        return samples , targets


