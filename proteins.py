from sklearn import svm
import csv
import numpy as np
from sklearn import cross_validation
import math
import random


test_size = 0.15
genotypes, treatments, behaviors = [], [], []
inputs = []
outputGenotypes, outputTreatments, outputBehaviors = [], [], []

with open('./proteins.csv', 'r') as csvfile:
	fireReader = csv.reader(csvfile)
	fireReader.next()
	for row in fireReader:
		try:
			proteins = map(lambda x: float(x) if x != '' else 0.0, row[1:-4])
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


genotype_train, genotype_test, genotypeOutput_train, genotypeOutput_test = cross_validation.train_test_split(inputs, outputGenotypes, test_size=test_size, random_state=0)
treatment_train, treatment_test, treatmentOutput_train, treatmentOutput_test = cross_validation.train_test_split(inputs, outputTreatments, test_size=test_size, random_state=0)
behavior_train, behavior_test, behaviorOutput_train, behaviourOutput_test = cross_validation.train_test_split(inputs, outputBehaviors, test_size=test_size, random_state=0)

genotypePredictor = svm.SVC().fit(genotype_train, genotypeOutput_train)
treatmentPredictor = svm.SVC().fit(treatment_train, treatmentOutput_train)
behaviorPredictor = svm.SVC().fit(behavior_train, behaviorOutput_train)

genotypeCorrect, genotypeIncorrect , genotypeFalsePositive, genotypeFalseNegative, totalGenotype = 0,0,0,0,0
for x, y in zip(genotype_test, genotypeOutput_test):
	predicted = genotypePredictor.predict(x)[0]
	if predicted == y:
		genotypeCorrect += 1
	else:
		genotypeIncorrect += 1
		if predicted == 1:
			genotypeFalsePositive += 1
		else:
			genotypeFalseNegative += 1
	totalGenotype += 1

print '\nGenotype Analysis : \n'
print 'Correct : {0:.2f} ({1:.2f} %)\nIncorrect : {2:.2f} ({3:.2f} %)\nFalse Positive : {4:.2f} ({5:.2f} %)\nFalse Negative : {6:.2f} ({7:.2f} %)'.format(genotypeCorrect, 100.0 * float(genotypeCorrect) / float(totalGenotype), genotypeIncorrect, 100.0 * float(genotypeIncorrect) / float(totalGenotype), genotypeFalsePositive, 100.0 * float(genotypeFalsePositive) / float(totalGenotype), genotypeFalseNegative, 100.0 * float(genotypeFalseNegative) / float(totalGenotype))


#This code was created and edited by Shelby Vanhooser