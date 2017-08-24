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
from sklearn.cluster import KMeans

eps = 1e-20

logger = formatted_logger('RelationalTopicModel', 'info')


class RelationalTopicModel:
    """ implementation of Gaussian Relational Topic Model
    I implemented the exponential link probability function in here

    Attributes
    ----------
    eta: ndarray, shape (n_topic)
        coefficient of exponential function
    rho: int
        pseudo number of negative example
    """

    def __init__(self, n_topic, n_doc, alpha=2, rho=10000, **kwargs):
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
        self.dim = 100
        self.u = np.zeros((self.n_topic,self.dim))
        self.Sigma = np.zeros((self.n_topic,self.dim,self.dim))
        self.log_norm_shift = 150

        self.doc_links = 0
        self.c = 100

        self.verbose = kwargs.pop('verbose', True)

        logger.info('Initialize RTM: num_topic:%d, num_doc:%d' % (self.n_topic, self.n_doc))

    def fit(self, doc_ids, doc_dir, doc_links, test_link, max_iter=100):
        # doc_ids is the str id list of users
        # doc_dir is the location of .mat files
        # doc_links is the links matrix
        # initialize topic mean and covariance using niw
        # default_stats_file = './data/data_stats.mat'
        # if os.path.exists(default_stats_file):
        #     data_stats = io.loadmat(default_stats_file)
        #     x_mean = data_stats['x_mean']
        #     x_mean = x_mean[0,:]
        #     S_x = data_stats['S_x']
        #     self.num_img = data_stats['num_img']
        # else:
        #     x_mean, num_img = self.get_data_mean(doc_ids, doc_dir)
        #     S_x = self.get_data_scatter(doc_ids, doc_dir, x_mean)
        #     data_stats = dict()
        #     data_stats['x_mean'] = x_mean
        #     data_stats['S_x'] = S_x
        #     data_stats['num_img'] = num_img
        #     self.num_img = num_img
        #     io.savemat(default_stats_file,data_stats)
        #
        # self.dim = S_x.shape[0]
        # S0 = np.diag(np.diag(S_x))/ self.num_img
        # m0 = x_mean
        # df0 = self.dim + 2
        # k0 = 0.01
        # K = self.n_topic
        # logger.info('Initialize u and Sigma...')
        #
        # default_niw_file = './data/niw.mat'
        # if os.path.exists(default_niw_file):
        #     data_stats = io.loadmat(default_niw_file)
        #     self.u = data_stats['u']
        #     self.Sigma = data_stats['Sigma']
        # else:
        #     self.u, self.Sigma = niw(m0, k0, df0, S0, K)
        #     data_stats = dict()
        #     data_stats['u'] = self.u
        #     data_stats['Sigma'] = self.Sigma
        #     io.savemat(default_niw_file,data_stats)

        # initialize u and Sigma by kmeans
        self.initialize_by_kmeans(doc_ids, doc_dir)

        self.post_process_mvn()

        self.doc_links = doc_links

        logger.info('Initialize phi...')
        for di in xrange(self.n_doc):
            data = io.loadmat(os.path.join(doc_dir,doc_ids[di]+'.mat'))
            images = data['images'][:,1:]
            Nd = images.shape[0]
            self.phi.append(np.random.dirichlet([10] * self.n_topic, Nd).T)  # list of KxW
            self.pi[di, :] = np.sum(self.phi[di], 1) / np.sum(self.phi[di])
            self.D_log_norm.append(np.zeros((1,1)))

        if self.verbose:
            elbo = self.compute_elbo(doc_ids, doc_links)
            auc_value, auc_pr = self.predict_train()
            auc_value_test, auc_pr_test = self.predict_test(test_link)
            logger.info('[ITER] %3d,\tELBO: %.3f\tAUC: %.3f\ttest AUC: %.3f\tAUC_PR: %.3f\ttest AUC_PR: %.3f', -1, elbo,auc_value, auc_value_test, auc_pr, auc_pr_test)
            self.log(-1)

        logger.info('Start Iteration...')
        for iter in xrange(max_iter):
            tic = time.time()
            (nu, eta, u, Sigma, phi, pi, gamma, D_log_norm) = (self.nu, self.eta.copy(), self.u.copy(), self.Sigma.copy(),
                list(self.phi), self.pi.copy(), self.gamma.copy(), list(self.D_log_norm))
            self.variation_update(doc_ids, doc_dir, doc_links)
            self.parameter_estimation(doc_links)

            self.print_diff(nu, eta, u, Sigma, phi, pi, gamma, D_log_norm)
            
            if self.verbose:
                elbo = self.compute_elbo(doc_ids, doc_links)
                auc_value, auc_pr = self.predict_train()
                auc_value_test, auc_pr_test = self.predict_test(test_link)
                logger.info('[ITER] %3d,\tElapsed time: %.3f\tELBO: %.3f\tAUC: %.3f\ttest AUC: %.3f\tAUC_PR: %.3f\ttest AUC_PR: %.3f',
                            iter, time.time()-tic, elbo, auc_value, auc_value_test, auc_pr, auc_pr_test)
                pr_train = self.recommend_train()
                pr_test = self.recommend_test(test_link)
                auc_value_full, auc_pr_full = self.predict_full(test_link)
                logger.info('[ITER] %3d,\tfull AUC: %.3f\tfull AUC_PR: %.3f\ttrain precision: %.3f\ttest precision: %.3f',iter, auc_value_full, auc_pr_full, pr_train,pr_test)

                self.log(iter)
        self.save_model('model')
    def print_diff(self, nu, eta, u, Sigma, phi, pi, gamma, D_log_norm):
        logger.info('nu_diff = %f\teta_diff = %f\tu_diff = %f\tSigma_diff = %f\tgamma_diff = %f',
            self.rel_error(nu,self.nu),self.rel_error(eta,self.eta),self.rel_error(u,self.u),self.rel_error(Sigma,self.Sigma),
            self.rel_error(gamma,self.gamma))
        phi_diff = 0
        for di in xrange(len(phi)):
            phi_diff = np.max([phi_diff, self.rel_error(np.array(phi[di]),np.array(self.phi[di]))])
        D_log_norm_diff = 0
        for di in xrange(len(D_log_norm)):
            D_log_norm_diff = np.max([D_log_norm_diff, self.rel_error(np.array(D_log_norm[di]),np.array(self.D_log_norm[di]))])
        logger.info('phi_diff = %f\tpi_diff = %f\tD_log_norm_diff = %f',
            phi_diff, self.rel_error(pi,self.pi), D_log_norm_diff)

    def initialize_by_kmeans(self, doc_ids, doc_dir):
        default_init_file = './data/kmeans.mat'
        if os.path.exists(default_init_file):
            logger.info('Loading u and Sigma for k topics.')
            data_stats = io.loadmat(default_init_file)
            self.u = data_stats['u']
            self.Sigma = data_stats['Sigma']
        else:
            logger.info('Doing kmeans to initialize u and Sigma for k topics.')
            all_images = []
            for di in xrange(self.n_doc):
                data = io.loadmat(os.path.join(doc_dir, doc_ids[di] + '.mat'))
                images = data['images'][:, 1:]
                all_images.append(images)
            all_images = np.vstack(all_images)
            dim = all_images.shape[1]
            kmeans_model = KMeans(n_clusters=self.n_topic, random_state=1).fit(all_images)
            labels = kmeans_model.labels_
            self.u = kmeans_model.cluster_centers_
            sigma = np.zeros((self.n_topic,dim,dim))
            for i in xrange(self.n_topic):
                cluster_images = all_images[labels == i,:]
                subn = cluster_images.shape[0]
                for j in xrange(subn):
                    sigma[i,:,:] += np.dot((cluster_images[j,:]-self.u[i])[:,np.newaxis],(cluster_images[j,:]-self.u[i])[:,np.newaxis].T)
                sigma[i,:,:] = sigma[i,:,:] / subn
            self.Sigma = sigma
            data_stats = dict()
            data_stats['u'] = self.u
            data_stats['Sigma'] = self.Sigma
            io.savemat(default_init_file,data_stats)
            # for i in xrange(self.n_topic):
            #     cluster_images = all_images[labels == i,:]
            #     subn = cluster_images.shape[0]
            #     probs = np.zeros(subn)
            #     for j in xrange(subn):
            #         probs[j] = stats.multivariate_normal.logpdf(cluster_images[j,:], mean=self.u[i], cov=self.Sigma[i])
            #     print(probs)
            # pass
    def rel_error(self,x,y):
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

    def get_data_mean(self, doc_ids, doc_dir):
        dim = 100
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
        dim = 100
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
        E_qw = 0
        E_qtheta = 0
        E_qz = 0
        H_theta=0
        H_z = 0
        for di in xrange(self.n_doc):
            #words = doc_ids[di]
            #cnt = doc_cnt[di]
            
            log_norm = self.D_log_norm[di]
            E_qw += np.sum( self.phi[di] * log_norm)  # E_q[log p(w_{d,n}|\beta,z_{d,n})]   
            E_qtheta += np.sum((self.alpha - 1.) * e_log_theta[di, :])  # E_q[log p(\theta_d | alpha)]
            E_qz += np.sum(self.phi[di].T * e_log_theta[di, :])  # E_q[log p(z_{d,n}|\theta_d)]
            H_theta += -gammaln(np.sum(self.gamma[di, :])) + np.sum(gammaln(self.gamma[di, :])) \
                    - np.sum((self.gamma[di, :] - 1.) * (e_log_theta[di, :]))  # - E_q[log q(theta|gamma)]
            H_z += - np.sum(self.phi[di] * np.log(self.phi[di] + eps))  # - E_q[log q(z|phi)]
            
        elbo += E_qw
        elbo += E_qtheta
        elbo += E_qz
        elbo += H_theta
        elbo += H_z

        link_loss = 0
        for di in xrange(self.n_doc):
            for i in xrange(self.n_doc):
                if doc_links[di,i] == 1:
                    link_loss += np.dot(self.eta,
                                    self.pi[di] * self.pi[i]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]
        link_loss /= 2.
        # maybe regularization term
        pass
        elbo += link_loss
        print('E_qw=%f,E_qtheta=%f,E_qz=%f,H_theta=%f,H_z=%f,link_loss=%f' % (E_qw,E_qtheta,E_qz,H_theta,H_z,link_loss))

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
                    log_norm[k,i] = self.log_normal(images[i,:],k)

            self.D_log_norm[di] = log_norm

            e_log_theta_di = e_log_theta[di, :][:, np.newaxis]
            new_phi = log_norm + e_log_theta_di

            gradient = np.zeros(self.n_topic)
            # for ai in doc_links[di]:
            #     gradient += self.eta * self.pi[ai, :] / doc_len
            for i in xrange(self.n_doc):
                if doc_links[di,i] == 1:
                    gradient += self.eta * self.pi[i, :] / doc_len

            new_phi += gradient[:, np.newaxis]
            new_phi = np.exp(new_phi  + self.log_norm_shift)
            new_phi = new_phi / (np.sum(new_phi, 0))

            self.phi[di] = new_phi

            self.pi[di, :] = np.sum(self.phi[di], 1) / (np.sum(self.phi[di]) + eps)
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

        new_u = new_u / (sum_phi[:, np.newaxis] + eps)
        for k in xrange(self.n_topic):
            new_sigma[k,:,:] = new_sigma[k,:,:]/(sum_phi[k] + eps) - np.dot(new_u[k,:][:,np.newaxis],new_u[k,:][:,np.newaxis].T)
        
        self.u = new_u
        self.Sigma = new_sigma
        #whenever update u and Sigma, update mvn_param
        self.post_process_mvn()
        #self.beta = new_beta / np.sum(new_beta, 1)[:, np.newaxis]

    def parameter_estimation(self, doc_links):
        # update eta, nu
        pi_sum = np.zeros(self.n_topic)

        num_links = 0.

        for di in xrange(self.n_doc):
            for i in xrange(self.n_doc):
                if doc_links[di,i] == 1:
                    pi_sum += self.pi[di, :] * self.pi[i, :]
                    num_links += 1

        num_links /= 2.  # divide by 2 for bidirectional edge
        pi_sum /= 2.

        pi_alpha = np.zeros(self.n_topic) + self.alpha / (self.alpha * self.n_topic) * self.alpha / (self.alpha * self.n_topic)

        self.nu = np.log(num_links - np.sum(pi_sum) + eps) - np.log(
            self.rho * (self.n_topic - 1) / self.n_topic + num_links - np.sum(pi_sum) + eps)
        self.eta = np.log(pi_sum + eps) - np.log(pi_sum + self.rho * pi_alpha + eps) - self.nu

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

        output = dict()
        output['y'] = np.array(y)
        output['scores'] = np.array(scores)
        io.savemat('result_train.mat',output)
        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        auc_value = metrics.auc(fpr, tpr)
        auc_pr = metrics.average_precision_score(y, scores)
        return (auc_value, auc_pr)
    def recommend_train(self):
        scores = np.zeros((self.n_doc,self.n_doc))
        for di in xrange(self.n_doc):
            for dj in xrange(di+1, self.n_doc):
                score = np.dot(self.eta,
                                    self.pi[di] * self.pi[dj]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]
                scores[di,dj] = score
                scores[dj,di] = score
        ind = np.argsort(scores,axis=1)[:,::-1]
        num_recommend = 5
        hit = np.zeros((self.n_doc,num_recommend))
        for di in xrange(self.n_doc):
            hit[di,:] = self.doc_links[di,ind[di,:num_recommend]]
        #hit = self.doc_links[:,ind[:,:num_recommend]]
        prec = np.sum(hit)/(self.n_doc*num_recommend)
        return prec


    def predict_test(self, test_links):
        # predict links between users inside the dataset
        scores = list()
        y = list()
        for di in xrange(self.n_doc):
            for dj in xrange(di+1, self.n_doc):
                if self.doc_links[di,dj] == 0:
                    score = np.dot(self.eta,
                                        self.pi[di] * self.pi[dj]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]
                    scores.append(score)
                    y.append(test_links[di,dj])
        output = dict()
        output['y'] = np.array(y)
        output['scores'] = np.array(scores)
        io.savemat('result_test.mat', output)
        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        auc_value = metrics.auc(fpr, tpr)
        auc_pr = metrics.average_precision_score(y, scores)
        return (auc_value,auc_pr)

    def predict_full(self, test_links):
        # predict links between users inside the dataset
        ground_truth = self.doc_links + test_links
        scores = list()
        y = list()
        for di in xrange(self.n_doc):
            for dj in xrange(di + 1, self.n_doc):
                score = np.dot(self.eta, self.pi[di] * self.pi[dj]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]
                scores.append(score)
                y.append(ground_truth[di, dj])
        output = dict()
        output['y'] = np.array(y)
        output['scores'] = np.array(scores)
        io.savemat('result_full.mat', output)
        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        auc_value = metrics.auc(fpr, tpr)
        auc_pr = metrics.average_precision_score(y, scores)
        return (auc_value, auc_pr)

    def recommend_test(self, test_links):
        scores = np.zeros((self.n_doc,self.n_doc))
        for di in xrange(self.n_doc):
            for dj in xrange(di+1, self.n_doc):
                score = np.dot(self.eta,
                                    self.pi[di] * self.pi[dj]) + self.nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]
                scores[di,dj] = score
                scores[dj,di] = score
        ind = np.argsort(scores, axis=1)[:, ::-1]
        num_recommend = 5
        ground_truth = self.doc_links + test_links
        hit = np.zeros((self.n_doc, num_recommend))
        for di in xrange(self.n_doc):
            hit[di, :] = ground_truth[di, ind[di, :num_recommend]]
            # hit = self.doc_links[:,ind[:,:num_recommend]]
        prec = np.sum(hit) / (self.n_doc * num_recommend)
        return prec

    def log(self,it):
        log_directory = 'log'
        if not os.path.exists(log_directory):
            os.mkdir(log_directory)

        output = {}
        output['eta'] = self.eta
        output['nu'] = self.nu
        output['u'] = self.u
        output['Sigma'] = self.Sigma
        output['gamma'] = self.gamma
        output['alpha'] = self.alpha
        phi = np.zeros((len(self.phi),), dtype=np.object)
        for i in xrange(len(self.phi)):
            phi[i] = self.phi[i]
        output['phi'] = phi
        output['pi'] = self.pi

        io.savemat(log_directory + '/' + str(it) + '.mat', output)

    def save_model(self, output_directory):
        
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        output = {}
        output['eta'] = self.eta
        output['nu'] = self.nu
        output['u'] = self.u
        output['Sigma'] = self.Sigma
        output['gamma'] = self.gamma
        output['alpha'] = self.alpha
        phi = np.zeros((len(self.phi),), dtype=np.object)
        for i in xrange(len(self.phi)):
            phi[i] = self.phi[i]
        output['phi'] = phi
        output['pi'] = self.pi
        output['n_doc'] = self.n_doc
        output['n_topic'] = self.n_topic
        D_log_norm = np.zeros((len(self.D_log_norm),), dtype=np.object)
        for i in xrange(len(self.D_log_norm)):
            D_log_norm[i] = self.D_log_norm[i]
        output['D_log_norm'] = D_log_norm
        output['rho'] = self.rho
        output['dim'] = self.dim
        output['log_norm_shift'] = self.log_norm_shift
        output['doc_links'] = self.doc_links

        io.savemat(output_directory + '/parameters.mat', output)


