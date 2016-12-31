import subprocess

a = "python ../RecommendationSystem/Probabilistic-matrix-factorization-in-Python/RunPMF.py --f "
for i in range(10):
    #subprocess.call("pwd")
    subprocess.call((a + str(10 * (i + 3))).split())
    subprocess.call("python dataUtils.py".split())
