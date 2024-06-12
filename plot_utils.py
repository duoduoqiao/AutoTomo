import matplotlib.pyplot as plt
import numpy as np


def sp_l1(true_test_flow, pred_test_flow):

    tn, fn = true_test_flow.shape 
    base = np.mean(true_test_flow, axis=0)
    zero_idx = np.where(base==0)
    base[zero_idx] = 1.
    nmae = np.mean(np.abs(true_test_flow-pred_test_flow), axis=0) / base
    return np.arange(fn), nmae


def tm_l1(true_test_flow, pred_test_flow, pat=12):
    
    tn, fn = true_test_flow.shape
    results = np.zeros((tn//pat))
    
    base = np.mean(true_test_flow, axis=1)
    zero_idx = np.where(base==0)
    base[zero_idx] = 1.
    nmae = np.mean(np.abs(true_test_flow-pred_test_flow), axis=1) / base
    for i in range(tn//pat):
        results[i] = np.sum(nmae[i*pat:(i+1)*pat])
    return np.arange(tn//pat), results


def plot_sp(real_flow, pred_flow, label, loss, pict_name):

    fig, ax = plt.subplots()
    x, f = sp_l1(real_flow, pred_flow)
    y_label = "SRE"
    
    ax.plot(x, f, label=label)
    ax.set_xlabel('FlowId', fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label,  fontsize=14, fontweight='bold')
    ax.set_title('Figure 1')
    ax.legend()
    plt.savefig(pict_name)
    plt.show()


def plot_tm(real_flow, pred_flow, label, loss, pict_name):

    fig, ax = plt.subplots()

    x, f = tm_l1(real_flow, pred_flow)
    y_label = "TRE"

    ax.plot(x, f, label=label)
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label,  fontsize=14, fontweight='bold')
    ax.set_title('Figure 2')
    ax.legend()
    plt.savefig(pict_name)
    plt.show()