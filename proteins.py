from sklearn.svm import SVC
import csv
import numpy as np
from sklearn import cross_validation
import math
import random


genotypes, treatments, behaviors = [], [], []
inputs = []
outputGenotypes, outputTreatments, outputBehaviors = [], [], []

with open('./proteins.csv', 'r') as csvfile:
	fireReader = csv.reader(csvfile)
	fireReader.next()
	for row in fireReader:
		try:
			proteins = map(float, row[1:-4])
			genotype, treatment, behavior = row[-4:-1]
			if not genotype in genotypes:
				genotypes += [genotype]
			if not treatment in treatments:
				treatments += [treatment]
			if not behavior in behaviors:
				behaviors += [behavior]

			inputs += [proteins]
			outputGenotypes += [genotypes.index(genotype)]
			outputTreatments += [treatments.index(treatment)]
			outputBehaviors += [behaviors.index(behavior)]
		except Exception, e:
			print 'Parser error', str(e)

genotypePredictor = SVC().fit(inputs, outputGenotypes)
treatmentPredictor = SVC().fit(inputs, outputTreatments)
behaviorPredictor = SVC().fit(inputs, outputBehaviors)


#This code was created and edited by Shelby Vanhooser