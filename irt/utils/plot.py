import matplotlib.pyplot as plt
import numpy as np

def plot_icc(model, fName, item = None):
    if type(item) == type(None):
        fig = plt.figure(dpi=250)
        ax = fig.add_subplot(1,1,1)

        s = np.std(model.theta)
        mn = np.min(model.theta) - 4 * s
        mx = np.max(model.theta) + 4 * s
        r = np.arange(mn, mx, s/10)

        for i, (a, b) in enumerate(zip(model.a, model.b)):
            ax.plot(r, model.logistic(a * (r - b)), label = "Item {}".format(i+1))

        ax.legend(loc = 0)
        fig.tight_layout()
        fig.savefig(fName)
        plt.close(fig)
    else:
        fig = plt.figure(dpi=250)
        ax = fig.add_subplot(1,1,1)

        s = np.std(model.theta)
        mn = np.min(model.theta) - 4 * s
        mx = np.max(model.theta) + 4 * s
        r = np.arange(mn, mx, s/10)

        a = model.a[item]
        b = model.b[item]
        ax.plot(r, model.logistic(a * (r - b)), label = "Item {}".format(item+1))
        ax.scatter(model.theta, model.data[:, item:item+1].reshape(model.data.shape[0]), lw = 0.1)

        ax.legend(loc = 0)
        fig.tight_layout()
        fig.savefig(fName)
        plt.close(fig)


def plot_info(model, fName):
    fig = plt.figure(dpi=250)
    ax = fig.add_subplot(1,1,1)

    s = np.std(model.theta)
    mn = np.min(model.theta) - 3 * s
    mx = np.max(model.theta) + 3 * s
    r = np.arange(mn, mx, s/10)

    info = np.vectorize(lambda a, b, theta: (a ** 2) * model.__p_uij__(1, a, b, theta) * model.__p_uij__(0, a, b, theta))
    def dTHdthk(model, th):
        tempth = model.theta[0]
        model.theta[0] = th
        result = model.__dTHdthk__(0)
        model.theta[0] = tempth
        return result
    info_T = np.vectorize(lambda th: -1 * dTHdthk(model, th))

    ax.plot(r, info_T(r), label = "Total Information")
    for i, (a, b) in enumerate(zip(model.a, model.b)):
        ax.plot(r, info(a, b, r), label = "Item {}".format(i+1), ls = "--")

    ax.legend(loc = 0)
    fig.tight_layout()
    fig.savefig(fName)
    plt.close(fig)