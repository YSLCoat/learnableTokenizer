import numpy as np

class cluster():
    def __init__(self, l, a, b, x, y):
        self.l = l
        self.a = a 
        self.b = b 
        self.x = x
        self.y = y
        
    
    def update(self, l, a, b, x, y):
        self.l = l
        self.a = a 
        self.b = b 
        self.x = x
        self.y = y
        
    def get_lab(self):
        return self.l, self.a, self.b
    
    def get_xy(self):
        return self.x, self.y
    
    def get_labxy(self):
        return self.l, self.a, self.b, self.x, self.y
    
    
class slic():
    def __init__(self, img, n_segments, compactness, max_iter, min_size):
        self.img = img
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size = min_size
        self.height, self.width, self.channels = img.shape
        self.step = int(np.sqrt(self.height * self.width / self.n_segments))
        self.clusters = []
        self.labels = np.full((self.height, self.width), -1)
        self.distances = np.full((self.height, self.width), np.inf)
        self.generate_clusters()
        self.iterate()
        self.create_labels()
        
    def generate_clusters(self):
        for i in range(self.step, self.height - self.step, self.step):
            for j in range(self.step, self.width - self.step, self.step):
                l, a, b = self.img[i][j]
                self.clusters.append(cluster(l, a, b, i, j))
                
    def iterate(self):
        for i in range(self.max_iter):
            for cluster in self.clusters:
                for i in range(cluster.x - self.step, cluster.x + self.step):
                    for j in range(cluster.y - self.step, cluster.y + self.step):
                        if i >= 0 and i < self.height and j >= 0 and j < self.width:
                            l, a, b = self.img[i][j]
                            d = self.distance(cluster, l, a, b, i, j)
                            if d < self.distances[i][j]:
                                self.distances[i][j] = d
                                self.labels[i][j] = cluster
            for cluster in self.clusters:
                l, a, b, x, y = 0, 0, 0, 0, 0
                count = 0
                for i in range(cluster.x - self.step, cluster.x + self.step):
                    for j in range(cluster.y - self.step, cluster.y + self.step):
                        if i >= 0 and i < self.height and j >= 0 and j < self.width:
                            if self.labels[i][j] == cluster:
                                tl, ta, tb = self.img[i][j]
                                l += tl
                                a += ta
                                b += tb
                                x += i
                                y += j
