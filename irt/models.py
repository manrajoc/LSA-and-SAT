import numpy as np
import math
from scipy.stats import norm, percentileofscore, zscore

class logistic_2p(object):
    def __init__(self, data): # Checked
        self.data = np.copy(data)
        self.a = np.ones(data.shape[1])
        # self.a = np.random.normal(2, 0.5, data.shape[1])
        self.b = self.__get_b_estimates__(data)
        self.theta = self.__get_theta_estimates__(data)
        self.a_log = [np.copy(self.a)]
        self.b_log = [np.copy(self.b)]
        self.theta_log = [np.copy(self.theta)]


    def NEWTON(self):
        def del_ab(item):
            H = self.__item_hessian__(item)
            G = np.array([[self.__A__(item)],
                          [self.__B__(item)]])

            return -1 * np.matmul(np.linalg.inv(H), G)
        del_a = np.vectorize(lambda i: del_ab(i)[0])
        del_b = np.vectorize(lambda i: del_ab(i)[1])
        del_th = np.vectorize(lambda j: -1 * np.reciprocal(self.__dTHdthk__(j)) * self.__TH__(j))
        nItems = self.data.shape[1]
        nSubs = self.data.shape[0]
        newtonCycles = 3

        for bCycle in range(5):
            for nCyc in range(newtonCycles):
                self.a = self.a + del_a(np.arange(nItems))
                self.b = self.b + del_b(np.arange(nItems))
            self.a_log.append(self.a)
            self.b_log.append(self.b)

            for nCyc in range(newtonCycles):
                self.theta = self.theta + del_th(np.arange(nSubs))
            self.theta_log.append(self.theta)


    def GA(self, lmbda, item_error = 1e-2, subject_error = 1e-1): # Checked
        gradA = np.vectorize(lambda i: self.__A__(i))
        gradB = np.vectorize(lambda i: self.__B__(i))
        gradTH = np.vectorize(lambda j: self.__TH__(j))
        item_index_vector = np.arange(self.data.shape[1])
        subject_index_vector = np.arange(self.data.shape[0])

        epoch = 0
        while True:
            error = item_error + 1
            while error >= (item_error/2):
                dela, delb = lmbda * gradA(item_index_vector), lmbda * gradB(item_index_vector)
                self.a = self.a + dela
                self.b = self.b + delb
                error = max(max(np.abs(dela)), max(np.abs(delb)))
            a_conv = np.max(np.abs(self.a - self.a_log[-1]))
            b_conv = np.max(np.abs(self.b - self.b_log[-1]))
            self.a_log.append(self.a)
            self.b_log.append(self.b)

            error = subject_error + 1
            while error >= (subject_error/2):
                deltheta = lmbda * gradTH(subject_index_vector)
                self.theta = self.theta + deltheta
                error = max(np.abs(deltheta))
            th_conv = np.max(np.abs(self.theta - self.theta_log[-1]))
            self.theta_log.append(self.theta)
            
            print("Birnbaum Iteration #{} Completed".format(epoch+1))
            epoch += 1
            if a_conv < item_error and b_conv < item_error and th_conv < subject_error:
                break

        
    def __get_b_estimates__(self, data): # Checked
        scores = np.sum(data, axis = 0)
        o_scores = sorted(scores)
        percentile = np.vectorize(lambda x: percentileofscore(o_scores, x, kind='weak'))
        p = percentile(scores) / 100
        e = norm.ppf(p, loc = 0, scale = 1)
        e[e == np.inf] = 0 + 5 * 1
        return e

    def __get_theta_estimates__(self, data): # Checked
        scores = np.sum(data, axis = 1)
        o_scores = sorted(scores)
        percentile = np.vectorize(lambda x: percentileofscore(o_scores, x, kind='weak'))
        p = percentile(scores) / 100
        e = norm.ppf(p, loc = 0, scale = 1)
        e[e == np.inf] = 0 + 5 * 1
        return e
    
    def __subject_hessian__(self): # Checked
        hessian = np.zeros(self.theta.shape)
        for j in range(self.data.shape[0]):
            hessian[j] = self.__dTHdthk__(j)
        return hessian

    def __item_hessian__(self, i): # Checked
        hessian = np.zeros((2,2))
        hessian[0][0] = self.__dAdak__(i)
        hessian[1][1] = self.__dBdbk__(i)
        hessian[0][1] = self.__dAdbk__(i)
        hessian[1][0] = self.__dBdak__(i)
        return hessian

    def __dTHdthk__(self, j): # Checked
        a, b, thetaj = self.a, self.b, self.theta[j]
        result = 0
        for i in range(self.data.shape[1]):
            k = a[i]
            k = k ** 2
            p = self.__p_uij__(1, a[i], b[i], thetaj)
            q = self.__p_uij__(0, a[i], b[i], thetaj)
            result += k * p * q
        return -1 * result

    def __dAdak__(self, i): # Checked
        ai, bi, theta = self.a[i], self.b[i], self.theta
        result = 0
        for j in range(self.data.shape[0]):
            k = theta[j] - bi
            k = k ** 2
            p = self.__p_uij__(1, ai, bi, theta[j])
            q = self.__p_uij__(0, ai, bi, theta[j])
            result += k * p * q
        return -1 * result
    
    def __dAdbk__(self, i): # Checked
        ui = self.data[:, i:i+1].reshape(self.data.shape[0])
        ai, bi, theta = self.a[i], self.b[i], self.theta
        result = 0
        for j, uij in enumerate(ui):
            z = ai * (theta[j] - bi)
            p = self.__p_uij__(1, ai, bi, theta[j])
            q = self.__p_uij__(0, ai, bi, theta[j])
            if uij:
                result += z * p * q - q
            else:
                result += z * p * q + p
        return result

    def __dBdak__(self, i): # Checked
        ui = self.data[:, i:i+1].reshape(self.data.shape[0])
        ai, bi, theta = self.a[i], self.b[i], self.theta
        result = 0
        for j, uij in enumerate(ui):
            z = ai * (theta[j] - bi)
            p = self.__p_uij__(1, ai, bi, theta[j])
            q = self.__p_uij__(0, ai, bi, theta[j])
            if uij:
                result += z * p * q - q
            else:
                result += z * p * q + p
        return result

    def __dBdbk__(self, i): # Checked
        ai, bi, theta = self.a[i], self.b[i], self.theta
        result = 0
        for j in range(self.data.shape[0]):
            k = ai
            k = k ** 2
            p = self.__p_uij__(1, ai, bi, theta[j])
            q = self.__p_uij__(0, ai, bi, theta[j])
            result += k * p * q
        return -1 * result

    def __A__(self, i): # Checked
        """ Returns the derivative of the log liklihood function with respect to ai """
        ui = self.data[:, i:i+1].reshape(self.data.shape[0])
        ai, bi, theta = self.a[i], self.b[i], self.theta
        result = 0
        for j, uij in enumerate(ui):
            k = theta[j] - bi
            if uij:
                result += k * self.__p_uij__(0, ai, bi, theta[j])
            else:
                result += -1 * k * self.__p_uij__(1, ai, bi, theta[j])
        return result
    
    def __B__(self, i): # Checked
        """ Returns the derivative of the log liklihood function with respect to bi """
        ui = self.data[:, i:i+1].reshape(self.data.shape[0])
        ai, bi, theta = self.a[i], self.b[i], self.theta
        result = 0
        for j, uij in enumerate(ui):
            k = -1 * ai
            if uij:
                result += k * self.__p_uij__(0, ai, bi, theta[j])
            else:
                result += -1 * k * self.__p_uij__(1, ai, bi, theta[j])
        return result

    def __TH__(self, j): # Checked
        """ Returns the derivative of the log liklihood function with respect to bi """
        uj = self.data[j:j+1, :].reshape(self.data.shape[1])
        a, b, thetaj = self.a, self.b, self.theta[j]
        result = 0
        for i, uij in enumerate(uj):
            k = a[i]
            if uij:
                result += k * self.__p_uij__(0, a[i], b[i], thetaj)
            else:
                result += -1 * k * self.__p_uij__(1, a[i], b[i], thetaj)
        return result

    def __dp_dai__(self, ai, bi, thetaj): # Checked
        """ Returns the derivative of p(u = 1 | ai, bi, thetaj) w.r.t. a """
        k = thetaj - bi
        z = ai * (thetaj - bi)
        p = self.logistic(z)
        q = self.logistic(-z)
        return p * q * k

    def __dp_dbi__(self, ai, bi, thetaj): # Checked
        """ Returns the derivative of p(u = 1 | ai, bi, thetaj) w.r.t. b """
        k = ai
        z = ai * (thetaj - bi)
        p = self.logistic(z)
        q = self.logistic(-z)
        return p * q * k
    
    
    def __log_l__(self): # Checked
        """ Returns the total log liklihood of data """
        data = self.data
        l = 0
        for j, uj in enumerate(data):
            for i, uij in enumerate(uj):
                l += math.log(self.__p_uij__(uij, self.a[i], self.b[i], self.theta[j]))
        return l
    
    def __p_uij__(self, uij, ai, bi, thetaj): # Checked
        """ Returns p(u | a, b, theta) """
        z = ai * (thetaj - bi)
        if uij:
            return self.logistic(z)
        else:
            return self.logistic(-z)

    def logistic(self, z): # Checked
        """ Returns the logistic value of deviate z """
        return 1/(1+np.exp(-z, dtype = np.longdouble))

    def __print_probs__(self): # Checked
        data = self.data
        result = np.ones(data.shape)
        for j, uj in enumerate(data):
            for i, uij in enumerate(uj):
                result[j][i] = self.__p_uij__(uij, self.a[i], self.b[i], self.theta[j])
        print(result)
        print("\nSum of log : ", np.sum(np.log(result)))
        print("log of products : ", np.log(np.product(result)), "\n")
