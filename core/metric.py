

from file_cache.utils.util_log import *


def get_weighted_fscore(y_pred, y_true, dic_):
    f_score = 0
    for i in range(12):
        if i in dic_:
            yt = y_true == i
            yp = y_pred == i
            tmp_score = f1_score(y_true=yt, y_pred= yp)
            f_score += dic_[i] * tmp_score
            logger.info(f'{i:02},{dic_[i]:0.4f},{tmp_score:0.4f},{dic_[i]*tmp_score:0.4f}')
        else:
            logger.warning(f'Mode#{i} is zero')
    logger.info(f_score)
    return f_score



from sklearn.metrics import f1_score
def lgb_f1_score_avg(y_hat, data, average):
    y_true = data.get_label()
    num_sample = len(y_true)
    #print(y_hat.shape, y_hat[:10])
    if average=='binary':
        y_hat = [0 if item<0.3 else 1 for item in y_hat]
        # y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
        # print(y_true.shape, y_hat.shape, y_true[:10], y_hat[:10])
        score = f1_score(y_true, y_hat, average=average)
        return 'f1', round(score, 4), True
    else:
        y_hat =  y_hat.reshape(-1, num_sample).T.argmax(axis=1)
        #y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
        #print(y_true.shape, y_hat.shape, y_true[:10], y_hat[:10])
        score = f1_score(y_true, y_hat, average=average)
        return 'f1', round(score, 4) , True


from functools import partial

lgb_f1_score = partial(lgb_f1_score_avg, average='weighted')

lgb_f1_score_bin = partial(lgb_f1_score_avg, average='binary')
