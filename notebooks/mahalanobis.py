import numpy
import sys

def read_vectors(file):
  vectors = []
  counts = []
  with open(file, "r") as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()
      if line:
        count, vector = line.split("\t")
        vectors.append([float(p) for p in vector.split()])
        counts.append(int(count))

  return vectors, counts

def is_pos_def(x):
  return numpy.all(numpy.linalg.eigvals(x) > 0)

vectors, counts = read_vectors(sys.argv[1])
covariance_matrix = numpy.cov(numpy.array(vectors).T)

print ("Covariance matrix ", covariance_matrix)
print ("Covariance matrix eigenvalues ", numpy.linalg.eigvals(covariance_matrix))
print ("Covariance matrix is positive definite ", is_pos_def(covariance_matrix))
print ("Covariance matrix rank ", numpy.linalg.matrix_rank(covariance_matrix))
print ("Vectors rank ", numpy.linalg.matrix_rank(numpy.array(vectors)))

