import numpy as np
import matplotlib.pyplot as plt

def Result(n_points, data_dir, file_name, n_files):
    result = [[] for i in range(n_points)]
    for j in range(n_files):
        local_res = np.genfromtxt(data_dir + str(j+1) + "/" + file_name)
        for p in range(n_points):
            result[p].append(local_res[p])
    r = np.array(result)
    return np.mean(r, axis=1)

def AllResultsForOneMethod(n_points, data_dir, method, n_files, result_dir, save_file=''):
    mean_dist = Result(n_points, data_dir, method + "_mean_dist", 9)
    cp = Result(n_points, data_dir, method + "_correct_pairs", 9)
    precision10 = Result(n_points, data_dir, method + "_precision10", 9)
    if save_file == '':
        save_location = method
    else:
        save_location = save_file
    np.savetxt(result_dir + save_location + "_mean_dist", mean_dist)
    np.savetxt(result_dir + save_location + "_correct_pairs", cp)
    np.savetxt(result_dir + save_location + "_precision10", precision10)

def Plot_All(data_dir, methods, metric):
    colors = ['g', 'r', 'c', 'm', 'y', 'k', 'w']
    shapes = [",", "o", "v", "^", "<", ">", "1"]
    fig = plt.figure()
    for i in range(len(methods)):
        data = np.genfromtxt(data_dir + methods[i] + "_" + metric)
        print(data.shape)
        plt.plot(np.arange(data.shape[0]-1), data[1:], colors[i] + shapes[i], 
                 np.arange(data.shape[0]-1), data[1:], 'k')
    plt.grid()
    fig.savefig(data_dir + metric + ".png")
