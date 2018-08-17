import scipy.io
import time
import sys
from collections import *
import numpy as np
import math
import copy
class model():
	def __init__(self, fileName, category, facets, epochs, alpha, beta, gama, m, power_var):
		self.fileName = fileName
		self.category = category
		self.facets = facets
		self.power_var = power_var
		self.epochs = epochs
		self.users_at_times = defaultdict(lambda: defaultdict(int))
		self.identity = np.zeros([self.facets, self.facets])
		for i in xrange(self.facets):
			self.identity[i][i] = beta
		self.m = m
		self.max_users = 0
		self.time_stamp = 0
		self.alpha = alpha
		self.beta = beta
		self.gama = gama
		self.trust_at_times = [[]]
		self.mat = []
		self.eta = []
		self.G = []
		self.c = []
		self.b = []
		self.a = []
		self.C = []
		self.B = []
		self.A = []
		self.U = []
		self.V = []
	def find_trust_at_times(self):
		self.mat = scipy.io.loadmat('epinion_trust_with_timestamp.mat')
		for i in self.mat[self.category]:
			if i[2] <= self.m:
				self.max_users = max(self.max_users, i[0], i[1])
				self.users_at_times[i[0]][i[1]] = i[2]
				self.time_stamp = max(self.time_stamp, i[2])
				try:
					self.trust_at_times[i[2]].append([i[0], i[1]])
				except:
					sz = len(self.trust_at_times[0])
					while sz <= i[2]:
						self.trust_at_times.append([])
						sz+=1
					self.trust_at_times[i[2]].append([i[0], i[1]])
		print self.max_users, self.time_stamp
	def printing_stuff(self, x, y):
		to_print = str(x) + '/' + str(y) + ' : ' + str(x/float(y))
		sys.stdout.write('\r{0}'.format(to_print))
		sys.stdout.flush()
		# time.sleep(0.001)
	def sigma(self,m):
		self.G = np.ones([self.max_users, self.max_users])
		for i in xrange(len(self.mat[self.category])):
			# print self.mat[self.category][i][2]
			if self.mat[self.category][i][2] <= m:
				self.G[self.mat[self.category][i][0]-1][self.mat[self.category][i][1]-1] = self.power_var
		self.G/=self.power_var
				# print self.mat[self.category][i][0]-1, self.mat[self.category][i][1]-1
	def find_matrix_c(self):
		self.c = np.zeros([self.max_users, self.max_users])
		temp_mat1 = np.matmul(self.U, np.matmul(self.V, self.U.T))
		self.c = (self.G - temp_mat1)**2
		# self.c = 
		# for i in xrange(self.max_users):
		# 	self.printing_stuff(i+1, self.max_users)
		# 	for j in xrange(self.max_users):
		# 		# m1 = self.U[i]
		# 		# m2 = np.array([self.U[j]]).T
		# 		# m3 = np.matmul(m1, np.matmul(self.V, m2))
		# 		self.c[i][j] = (self.G[i][j] - temp_mat1[i][j])**2
	def find_matrix_b(self,m):
		# print 'Starting B'
		self.b = np.ones([self.max_users, self.max_users])
		for i in xrange(self.max_users):
			# self.printing_stuff(i+1, self.max_users)
			if self.mat[self.category][i][2] <= m:
				u1 = self.mat[self.category][i][0]
				u2 = self.mat[self.category][i][1]
				t_first = self.users_at_times[u1][u2]
				sub = max(0, m - t_first)
				self.b[u1-1][u2-1] = pow(math.e, -self.eta[u1-1]*sub)
				# print u1-1, u2-1
	def find_matrix_a(self):
		# self.a = np.zeros([self.max_users, self.facets])
		temp_mat1 = self.b*self.G
		temp_mat2 = np.matmul(self.V, self.U.T).T
		temp_mat3 = np.matmul(self.V.T, self.U.T).T
		# print temp_mat2
		# print temp_mat3
		t1 = np.matmul(temp_mat1,temp_mat2)
		t2 = np.matmul(temp_mat1.T,temp_mat3)
		# print temp_mat2.shape
		# su = 0
		# for i in xrange(self.max_users):
		# 	su += temp_mat1[i].reshape(1,self.max_users).tolist()[0].count(0)
		# print su
		# print [sum(temp_mat1[i].reshape(1,self.max_users).tolist()[0].count(0)) for i in xrange(self.max_users)]
		# print temp_mat2.T[0].reshape(self.max_users,1).tolist()[0].count(0)
		# print np.matmul(temp_mat1[0].reshape(1,self.max_users), temp_mat2.T[0].reshape(self.max_users,1))
		# print temp_mat1.reshape(1,72573361)[0].tolist().count(0)
		self.a = (t1 + t2)
		# print self.a[5]
		# print self.a.shape
		# for i in xrange(self.max_users):
		# 	self.printing_stuff(i+1, self.max_users)
		# 	# t1, t2 = np.zeros(self.facets), np.zeros(self.facets)
		# 	for j in xrange(self.max_users):
		# 		# p1 = np.array([self.U[j]]).T
		# 		# print (self.b[i][j]*self.G[i][j]*self.VUT[j]).shape
		# 		# print self.a[i].shape, (self.b[i][j]*self.G[i][j]*self.UVT[j]).T, (self.b[j][i]*self.G[j][i]*self.VTUT[j]).T
		# 		self.a[i] += self.b[i][j]*self.G[i][j]*self.UVT[j]
		# 		self.a[i] += self.b[j][i]*self.G[j][i]*self.UV[j]
			# print np.array([t1 + t2]).T.shape
			# m1 = np.array([t1 + t2]).T
			# print t1.shape
			# print t1
			# self.a[i] = t1 + t2
	def find_matrix_A(self):
		self.A = np.zeros([self.max_users, self.facets, self.facets])
		for i in xrange(self.max_users):
			self.A[i] = self.identity
		temp_mat1 = []
		for i in xrange(self.max_users):
			temp_mat1.append(np.matmul(self.U[i].reshape(1, self.facets).T, self.U[i].reshape(1, self.facets)))
		temp_mat1 = np.array(temp_mat1)
		temp_mat2 = temp_mat1.view().reshape(self.max_users, self.facets**2)
		temp_mat3 = np.matmul(self.b, temp_mat2)
		temp_mat5 = np.matmul(self.b.T, temp_mat2)
		temp_mat4 = temp_mat3.view().reshape(self.max_users, self.facets, self.facets)
		temp_mat6 = temp_mat5.view().reshape(self.max_users, self.facets, self.facets)
		for i in xrange(self.max_users):
			# self.printing_stuff(i+1, self.max_users)
			self.A[i] = np.matmul(self.V, np.matmul(temp_mat4[i], self.V.T)) + np.matmul(self.V.T, np.matmul(temp_mat6[i], self.V))

		# self.A = np.zeros([self.max_users, self.facets, self.facets])
		# for i in xrange(self.max_users):
		# 	self.printing_stuff(i+1, self.max_users)
		# 	t1 = np.zeros([self.facets, self.facets])
		# 	for j in xrange(self.facets):
		# 		self.A[i][j][j] = self.beta
		# 	for j in xrange(self.max_users):
		# 		temp_mat1 = self.VUT[j]
		# 		# print [self.UVT[j]]
		# 		temp_mat2 = self.UVT[j].reshape(1, self.facets)
		# 		temp_mat3 = self.VTUT[j]
		# 		temp_mat4 = self.UV[j].reshape(1, self.facets)
		# 		# temp_mat1 = np.matmul(self.V, np.array([self.U[j]]).T)
		# 		# temp_mat2 = np.matmul(self.U[j], self.V.T)
		# 		# temp_mat3 = np.matmul(self.V.T, np.array([self.U[j]]).T)
		# 		# temp_mat4 = np.matmul(self.U[j], self.V)
		# 		temp_mat5 = np.matmul(temp_mat1, temp_mat2)
		# 		# temp_mat4 = np.array([temp_mat4])
		# 		temp_mat6 = np.matmul(temp_mat3, temp_mat4)
		# 		t1 = self.b[i][j]*temp_mat5
		# 		t2 = self.b[j][i]*temp_mat6
		# 		self.A[i] += t1 + t2
		# t1 = self.b*temp_mat5
		# t2 = (self.b*temp_mat6).T
		# print t1.shape
		# self.A += t1 + t2
		# for i in xrange(self.max_users):
		# 	t1, t2 = 0, 0
		# 	self.printing_stuff(i+1, self.max_users)
		# 	for j in xrange(self.max_users):
		# 		p1 = np.array([self.U[j]]).T
		# 		p2 = self.U[j]
		# 		p3 = np.matmul(self.V, p1)
		# 		p4 = np.matmul(p3, p2)
		# 		p5 = np.matmul(p4, self.V.T)
		# 		p6 = np.matmul(self.V.T, p1)
		# 		p7 = np.matmul(p6, p2)
		# 		p8 = np.matmul(p7, self.V)
		# 		t1 += self.b[i][j]*p5
		# 		t2 += self.b[j][i]*p8
		# 	self.A[i] += t1 + t2
	def find_matrix_B(self):
		# self.B = np.zeros([self.facets, self.facets])
		self.B = np.matmul(self.U.T, np.matmul(self.b*self.G, self.U))
		# print (self.b*self.G).reshape(1,72573361)[0].tolist().count(0)
		# temp_mat1 = self.b*self.G
		# temp_mat2 = np.matmul(self.U.T, self.U)
		# self.B = temp_mat1*temp_mat2
		# for i in xrange(self.max_users):
		# 	self.printing_stuff(i+1, self.max_users)
		# 	p1 = np.array([self.U[i]]).T
		# 	temp = np.zeros(self.facets)
		# 	for j in xrange(self.max_users):
		# 		# print j
		# 		temp += self.b[i][j]*self.G[i][j]*self.U[j]
		# 	self.B += p1*temp
	def find_matrix_C(self):
		# self.C = np.zeros([self.max_users, self.max_users])
		self.C = self.gama*self.V
		temp_mat1 = []
		for i in xrange(self.max_users):
			temp_mat1.append(np.matmul(self.U[i].reshape(1,self.facets).T, self.U[i].reshape(1,self.facets)))
		temp_mat1 = np.array(temp_mat1)
		temp_mat2 = temp_mat1.reshape(self.max_users, self.facets*self.facets)
		for i in xrange(self.max_users):
			temp_mat3 = np.matmul(self.b[i].reshape(1,self.max_users), temp_mat2).reshape(self.facets, self.facets)
			self.C += np.matmul(np.matmul(np.matmul(self.V.T, self.U[i].reshape(1,self.facets).T), self.U[i].reshape(1,self.facets)).T, temp_mat3)
		# temp_mat1 = np.matmul(self.U.T, self.U)
		# temp_mat2 = np.matmul(temp_mat1, self.V.T)
		# temp_mat3 = np.matmul(temp_mat2, temp_mat1)
		# temp_mat4 = self.b
		# for i in xrange(self.max_users):
		# 	self.printing_stuff(i+1, self.max_users)
		# 	for j in xrange(self.max_users):
		# 		p1 = np.matmul(self.U[i].reshape(1,self.facets).T, self.U[j].reshape(1,self.facets))
		# 		p2 = np.matmul(p1, self.V)
		# 		p3 = np.matmul(p2, p1)
		# 		self.C += self.b[i][j]*p3
	# def precompute(self):
	# 	self.VUT = np.zeros([self.max_users, self.facets, 1])
	# 	self.VTUT = np.zeros([self.max_users, self.facets, 1])
	# 	self.UV = np.zeros([self.max_users, self.facets])
	# 	self.UVT = np.zeros([self.max_users, self.facets])
	# 	for i in xrange(self.max_users):
	# 		self.printing_stuff(1+i, self.max_users)
	# 		p2 = np.array([self.U[i]]).T
	# 		# print p2.shape, np.matmul(self.V.T, p2).shape
	# 		self.VUT[i] = np.matmul(self.V, p2)
	# 		self.VTUT[i] = np.matmul(self.V.T, p2)
	# 		self.UV[i] = np.matmul(self.U[i], self.V)
	# 		self.UVT[i] = np.matmul(self.U[i], self.V.T)
			# self.UTU[i] = np.matmul(p2, self.U[i].reshape(1,self.facets))
	def model_train(self, m):
		self.eta = np.random.uniform(1,2,self.max_users)
		self.U = np.random.uniform(1,2,[self.max_users, self.facets])
		# print self.U
		self.V = np.random.uniform(1,2,[self.facets, self.facets])
		# print m
		self.sigma(m)
		# print 'Started b'
		# time_started = time.time()
		self.find_matrix_b(m)
		# print '\nTime taken for b = ', time.time() - time_started
		for i in xrange(self.epochs):
			print 'epochs #', i
			temp1 = np.zeros([self.max_users, self.facets])
			temp2 = np.zeros([self.facets, self.facets])
			time_started = time.time()
			# print 'Started precompute'
			# self.precompute()
			# print '\nTime taken for precompute = ', time.time() - time_started
			time_started = time.time()
			print 'Started c'
			self.find_matrix_c()
			print '\nTime taken for c = ', time.time() - time_started
			time_started = time.time()
			print 'Started a'
			self.find_matrix_a()
			print '\nTime taken for a = ', time.time() - time_started
			time_started = time.time()
			print 'Started A'
			self.find_matrix_A()
			print '\nTime taken for A = ', time.time() - time_started
			time_started = time.time()
			print 'Started B'
			self.find_matrix_B()
			print '\nTime taken for B = ', time.time() - time_started
			time_started = time.time()
			print 'Started C'
			self.find_matrix_C()
			print '\nTime taken for C = ', time.time() - time_started
			# print self.B
			# print self.C
			# time_started = time.time()
			print 'Updating eta'
			for j in xrange(self.max_users):
				num, den = 0, 0
				self.printing_stuff(j+1, self.max_users)
				for k in xrange(self.max_users):
					num += m*self.c[j][k]*self.b[j][k]
					den += self.users_at_times[j][k]*self.c[j][k]*self.b[j][k]
				# print num, den + 2*self.alpha*self.eta[j], self.eta[j]
				var = math.sqrt(num/(den + 2*self.alpha*self.eta[j]))
				self.eta[j] *= var
			print '\nTime taken for eta = ', time.time() - time_started
			time_started = time.time()
			print 'Updating U'
			for j in xrange(self.max_users):
				for k in xrange(self.facets):
					# print j,k
					num = self.a[j][k]
					# print self.U[j]
					den = np.matmul(self.U[j],self.A[j])
					# print num, den
					var = math.sqrt(num/den[k])
					temp1[j][k] = self.U[j][k]*var
			print '\nTime taken for U = ', time.time() - time_started
			time_started = time.time()
			print 'Updating V'
			for j in xrange(self.facets):
				for k in xrange(self.facets):
					# print j, k, self.B, self.C
					num = self.B[j][k]
					den = self.C[j][k]
					# print num, den
					var = math.sqrt(num/den)
					temp2[j][k] = self.V[j][k]*var
			# print temp2
			self.U = copy.deepcopy(temp1)
			self.V = copy.deepcopy(temp2)
			# print self.B
			# print self.C
			# print self.V
			# print self.U
			print '\nTime taken for V = ', time.time() - time_started
			time_started = time.time()
		G_new = np.matmul(self.U, np.matmul(self.V,self.U.T))
		scipy.io.savemat('predicted_out.mat', {'trust':G_new})		
		return G_new
