import numpy as np

def init_data(user_file, link_file):
	userList = list()
	userInd = dict()
	ind = 0
	for line in open(user_file):
		user = line.strip()
		userList.append(user)
		userInd[user] = ind
		ind += 1
	noOfUser = len(userList)
	link_matrix = np.zeros((noOfUser,noOfUser))

	for line in open(link_file):
		user1, user2 = line.strip().split(',')
		link_matrix[userInd[user1],userInd[user2]] = 1
		link_matrix[userInd[user2],userInd[user1]] = 1

	return (userList,link_matrix)

def split_train_test(user_file, link_file,ratio):
	userFdlist = list()
	for line in open(link_file):
		user1, user2 = line.strip().split(',')
		userFdlist.append((user1,user2))
	noOfLinks = len(userFdlist)
	ind = np.random.permutation(noOfLinks)
	noOfTrain = int(noOfLinks * ratio)
	fid = open('data/train.txt','w')
	for i in xrange(noOfTrain):
		fid.write('%s,%s\n' % (userFdlist[ind[i]]))
	fid.close()
	fid = open('data/test.txt','w')
	for i in xrange(noOfTrain,noOfLinks):
		fid.write('%s,%s\n' % (userFdlist[ind[i]]))
	fid.close()

