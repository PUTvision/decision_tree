import scipy.io

if __name__ == "__main__":
    mat = scipy.io.loadmat("..\\data\\terrain_data\\Kopia pociete_kroki_rasp.mat")
    print(mat["grs"])
    print(mat)
