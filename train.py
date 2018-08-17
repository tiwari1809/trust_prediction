from init_data import *
from multiprocessing import Pool
obj = model('epinion_trust_with_timestamp.mat', 'trust', 2, 10, 1, 1, 1, 2, 10**6)
obj.find_trust_at_times()
obj.model_train(3)
# pool = Pool()
# y_parallel = pool.map(obj.model_train(), 3) 
# # p.start()
# # p.join()