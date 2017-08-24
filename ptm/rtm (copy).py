from __future__ import print_function
import time

import numpy as np
from scipy.special import gammaln, psi
import scipy.stats as stats
import os
import scipy.io as io
from .progress import Progress
from six.moves import xrange

from sklearn import metrics

from .utils import write_top_words
from .formatted_logger import formatted_logger

from .niw import niw
import ptm.multivariate_normal as mvn

eps = 1e-20

logger = formatted_logger('RelationalTopicModel', 'info')


class RelationalTopicModel:
    """ implementation of relational topic model by Chang and Blei (2009)
    I implemented the exponential link probability function in here

    Attributes
    ----------
    eta: ndarray, shape (n_topic)
        coefficient of exponential function
    rho: int
        pseudo number of negative example
    """

    def __init__(self, n_topic, n_doc, alpha=0.1, rho=1000, **kwargs):
        self.n_doc = n_doc
        self.n_topic = n_topic

        self.alpha = alpha

        self.gamma = np.random.gamma(100., 1. / 100, [self.n_doc, self.n_topic])
        #self.beta = np.random.dirichlet([5] * self.n_voca, self.n_topic)
        self.D_log_norm = list()
        self.num_img = 0
        
        self.nu = 0
        self.eta = np.random.normal(0., 1, self.n_topic)

        self.phi = list()
        self.pi = np.zeros([self.n_doc, self.n_topic])

        self.rho = rho

        self.verbose = kwargs.pop('verbose', True)

        logger.info('Initialize RTM: num_topic:%d, num_doc:%d' % (self.n_topic, self.n_doc))

    def fit(self, doc_ids, doc_dir, doc_links, max_iter=100):
        # doc_ids is the str id list of users
        # doc_dir is the location of .mat files
        # doc_links is the links matrix
        # initialize topic mean and covariance using niw
        default_stats_file = './data/data_stats.mat'
        if os.path.exists(default_stats_file):
            data_stats = io.loadmat(default_stats_file)
            x_mean = data_stats['x_mean']
            x_mean = x_mean[0,:]
            S_x = data_stats['S_x']
            self.num_img = data_stats['num_img']
        else:
            x_mean, num_img = self.get_data_mean(doc_ids, doc_dir)
            S_x = self.get_data_scatter(doc_ids, doc_dir, x_mean)
            data_stats = dict()
            data_stats['x_mean'] = x_mean
            data_stats['S_x'] = S_x
            data_stats['num_img'] = num_img
            self.num_img = num_img
            io.savemat(default_stats_file,data_stats)

        self.dim = S_x.shape[0]
        S0 = np.diag(np.diag(S_x))/ self.num_img
        m0 = x_mean
        df0 = self.dim + 2
        k0 = 0.01
        K = self.n_topic
        logger.info('Initialize u and Sigma...')

        default_niw_file = './data/niw.mat'
        if os.path.exists(default_niw_file):
            data_stats = io.loadmat(default_niw_file)
            self.u = data_stats['u']
            self.Sigma = data_stats['Sigma']
        else:
            self.u, self.Sigma = niw(m0, k0, df0, S0, K)
            data_stats = dict()
            data_stats['u'] = self.u
            data_stats['Sigma'] = self.Sigma
            io.savemat(default_niw_file,data_stats)

        post_process_mvn()

        self.doc_links = doc_links

        logger.info('Initialize phi...')
        for di in xrange(self.n_doc):
            data = io.loadmat(os.path.join(doc_dir,doc_ids[di]+'.mat'))
            images = data['images'][:,1:]
            Nd = images.shape[0]
            self.phi.append(np.random.dirichlet([10] * self.n_topic, Nd).T)  # list of KxW
            self.pi[di, :] = np.sum(self.phi[di], 1) / np.sum(self.phi[di])
            self.D_log_norm.append(np.zeros((1,1)))

        logger.info('Start Iteration...')
        for iter in xrange(max_iter):
            tic = time.time()
            self.variation_update(doc_ids, doc_dir, doc_links)
            self.parameter_estimation(doc_links)
            if self.verbose:
                elbo = self.compute_elbo(doc_ids, doc_links)
                auc_value = self.predict_train()
                logger.info('[ITER] %3d,\tElapsed time: %.3f\tELBO: %.3f\tAUC: %.3f', iter, time.time()-tic, elbo, auc_value)
                

    def get_data_mean(self, doc_ids, doc_dir):
        dim = 1024
        x_mean = np.zeros(dim)
        num_images = 0
        logger.info('Getting data mean...')
        prog = Progress(len(doc_ids))
        for doc in doc_ids:
            data = io.loadmat(os.path.join(doc_dir,doc+'.mat'))
            images = data['images'][:,1:]
            num_images += images.shape[0]
            x_mean += np.sum(images,0)

            prog.update()
        prog.end()
        x_mean = x_mean / num_images

        return (x_mean, num_images)

    def get_data_scatter(self, doc_ids, doc_dir, x_mean):
        dim = 1024
        S_x = np.zeros((dim,dim))
        logger.info('Getting data scatter matrix...')
        prog = Progress(len(doc_ids))
        for doc in doc_ids:
            data = io.loadmat(os.path.join(doc_dir,doc+'.mat'))
            images = data['images'][:,1:]
            num_images = images.shape[0]
            for i in xrange(num_images):
                S_x += np.dot((images[i,:]-x_mean)[:,np.newaxis],(images[i,:]-x_mean)[:,np.newaxis].T)
            prog.update()
        prog.end()
        return S_x
        
    def post_process_mvn(self):
        self.mvn_param = list()
        for k in xrange(self.n_topic):
            dim, mean, cov = mvn._process_parameters(None, self.u[k], self.Sigma[k])
            prec_U, log_det_cov = mvn._psd_pinv_decomposed_log_pdet(cov)
            self.mvn_param.append((dim, mean, prec_U, log_det_cov))

    def log_normal(self, x, k):
        dim, mean,prec_U, log_det_cov = self.mvn_param[k]
        x = mvn._process_quantiles(x, dim)
        out = mvn._logpdf(x, mean, prec_U, log_det_cov)
        return mvn._squeeze_output(out)

    def compute_elbo(self, doc_ids, doc_links):
        """ compute evidence lower bound for trained model
        """
        elbo = 0

        e_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:, np.newaxis]  # D x K
        #log_beta = np.log(self.beta + eps)

        for di in xrange(self.n_doc):
            #words = doc_ids[di]
            #cnt = doc_cnt[di]
            
            log_norm = self.D_log_norm[di]
            elbo += np.sum( self.phi[di] * log_norm)  # E_q[log p(w_{d,n}|\beta,z_{d,n})]
            elbo += np.sum((self.alpha - 1.) * e_log_theta[di, :])  # E_q[log p(\theta_d | alpha)]
            elbo += np.sum(self.phi[di].T * e_log_theta[di, :])  # E_q[log p(z_{d,n}|\theta_d)]

            elbo += -gammaln(np.sum(self.gamma[di, :])) + np.sum(gammaln(self.gamma[di, :])) \
                    - np.sum((self.gamma[di, :] - 1.) * (e_log_theta[di, :]))  # - E_q[log q(theta|gamma)]
            elbo += - np.sum(self.phi[di] * np.log(self.phi[di]))  # - E_q[log q(z|phi)]

        link_loss = 0
        for di in xrange(self.n_doc):
            for adi in doc_links[di]:
                link_loss += np.dot(self.eta,
                                self.pi[di] * self.pi[adi]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]
        link_loss /= 2.
        # maybe regularization term
        pass
        elbo += link_loss

        return elbo

    def variation_update(self, doc_ids, doc_dir, doc_links):
        # update phi, gamma
        e_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, 1))[:, np.newaxis]

        #new_beta = np.zeros([self.n_topic, self.n_voca])
        sum_phi = 0
        new_u = np.zeros([self.n_topic, self.dim])
        new_sigma = np.zeros(self.Sigma.shape)
        prog = Progress(self.n_doc)
        for di in xrange(self.n_doc):
            #words = doc_ids[di]
            #cnt = doc_cnt[di]
            #doc_len = np.sum(cnt)
            data = io.loadmat(os.path.join(doc_dir,doc_ids[di]+'.mat'))
            images = data['images'][:,1:]
            doc_len = images.shape[0]
            #compute logN(im|uk,sk) for each image
            Nd = images.shape[0]
            log_norm = np.zeros((self.n_topic, Nd))
            for i in xrange(Nd):
                for k in xrange(self.n_topic):
                    log_norm[k,i] = stats.multivariate_normal.logpdf(images[i,:],self.u[k,:],self.Sigma[k,:,:])

            self.D_log_norm[di] = log_norm

            new_phi = log_norm + e_log_theta[di, :][:, np.newaxis]

            gradient = np.zeros(self.n_topic)
            for ai in doc_links[di]:
                gradient += self.eta * self.pi[ai, :] / doc_len

            new_phi += gradient[:, np.newaxis]
            new_phi = np.exp(new_phi)
            new_phi = new_phi / np.sum(new_phi, 0)

            self.phi[di] = new_phi

            self.pi[di, :] = np.sum(self.phi[di], 1) / np.sum(self.phi[di])
            self.gamma[di, :] = np.sum(self.phi[di], 1) + self.alpha
            
            #new_beta[:, words] += (cnt * self.phi[di])
            #update u
            sum_phi += np.sum(new_phi,1)
            for i in xrange(self.n_topic):
                new_u[i,:] += np.sum(new_phi[i,:][:, np.newaxis] * images,0)
            for i in xrange(Nd):
                for k in xrange(self.n_topic):
                    new_sigma[k,:,:] += new_phi[k,i] * np.dot(images[i,:][:,np.newaxis],images[i,:][:,np.newaxis].T)
            
            prog.update()
        prog.end()

        new_u = new_u / sum_phi[:, np.newaxis]
        for k in xrange(self.n_topic):
            new_sigma[k,:,:] = new_sigma[k,:,:]/sum_phi[k] - np.dot(new_u[k,:][:,np.newaxis],new_u[k,:][:,np.newaxis].T)
        
        self.u = new_u
        self.Sigma = new_sigma
        #self.beta = new_beta / np.sum(new_beta, 1)[:, np.newaxis]

    def parameter_estimation(self, doc_links):
        # update eta, nu
        pi_sum = np.zeros(self.n_topic)

        num_links = 0.

        for di in xrange(self.n_doc):
            for adi in doc_links[di]:
                pi_sum += self.pi[di, :] * self.pi[adi, :]
                num_links += 1

        num_links /= 2.  # divide by 2 for bidirectional edge
        pi_sum /= 2.

        pi_alpha = np.zeros(self.n_topic) + self.alpha / (self.alpha * self.n_topic) * self.alpha / (self.alpha * self.n_topic)

        self.nu = np.log(num_links - np.sum(pi_sum)) - np.log(
            self.rho * (self.n_topic - 1) / self.n_topic + num_links - np.sum(pi_sum))
        self.eta = np.log(pi_sum) - np.log(pi_sum + self.rho * pi_alpha) - self.nu

    def predict_train(self):
        # prediction of the training set
        scores = list()
        y = list()
        for di in xrange(self.n_doc):
            for dj in xrange(di+1, self.n_doc):
                score = np.dot(self.eta,
                                    self.pi[di] * self.pi[dj]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]
                scores.append(score)
                y.append(self.doc_links[di,dj])

        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        auc_value = metrics.auc(false_positive_rate, true_positive_rate)
        return auc_value

    def predict(self, test_links):
        # predict links between users inside the dataset
        scores = list()
        y = list()
        for di in xrange(self.n_doc):
            for dj in xrange(di+1, self.n_doc):
                if self.doc_links == 0:
                    score = np.dot(self.eta,
                                        self.pi[di] * self.pi[dj]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]
                    scores.append(score)
                    y.append(test_links[di,dj])

        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        auc_value = metrics.auc(false_positive_rate, true_positive_rate)
        return auc_value

    def save_model(self, output_directory, vocab=None):
        
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        output = {}
        output['eta'] = self.eta
        output['nu'] = self.nu
        output['u'] = self.u
        output['Sigma'] = self.Sigma
        output['gamma'] = self.gamma
        output['alpha'] = self.alpha
        output['phi'] = self.phi
        output['pi'] = self.pi

        io.savemat(output_directory + '/parameters.mat', output)
        # np.savetxt(output_directory + '/beta.txt', self.beta, delimiter='\t')
        # np.savetxt(output_directory + '/gamma.txt', self.gamma, delimiter='\t')
        # with open(output_directory + '/nu.txt', 'w') as f:
        #     f.write('%f\n' % self.nu)

        # if vocab is not None:
        #     write_top_words(self.beta, vocab, output_directory + '/top_words.csv')

