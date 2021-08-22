
import matplotlib.pyplot as plt

def vis_coef(lambdas, coefs, method=''):
    plt.figure(figsize=(8,5))
    ax = plt.gca()
    ax.plot(lambdas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.axis('tight')
    plt.xlabel('lambda', fontsize=12)
    plt.ylabel('weights', fontsize=12)
    plt.title('{} coefficients as a function of the regularization'.format(method), fontsize=14)
    plt.show()
    
def vis_mse(lambdas, MSE_set, best_lambda, best_mse, data='Validation'):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(lambdas, MSE_set)
    ax.scatter(best_lambda, best_mse, s=100, c='r', marker='x')
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    ax.text(x=best_lambda, y=best_mse+1000,s='MSE : {:.2f}'.format(best_mse))
    plt.axis('tight')
    plt.xlabel('lambda')
    plt.ylabel('Mean Squared Error')
    plt.title('{} set MSE'.format(data))
    plt.show()
    
    
def plotPR(score,fpr, tpr):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr,'r')
    plt.plot(fpr,1-fpr,'--')
    plt.text(0.1,0.2,'PR score : %.3f' %(score))
    plt.rc('font',size=13)
    plt.title('PR curve', fontsize=20)
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    
    
def plotROC(area,fpr, tpr):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr,'r')
    plt.plot(fpr,fpr,'--')
    plt.text(0.6,0.2,'AUROC : %.3f' %(area))
    plt.rc('font',size=13)
    plt.title('ROC curve', fontsize=20)
    plt.xlabel('False positive rates', fontsize=15)
    plt.ylabel('True positive rates', fontsize=15)
    plt.show()