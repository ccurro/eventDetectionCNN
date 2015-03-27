require 'nn'
require 'cutorch'
require 'cunn'
require 'audio'
require 'signal'
require 'gnuplot'
require 'torchx'

soundsPath  = '/afs/ee.cooper.edu/courses/ece412/eventDetectionData/singlesounds/sounds'

labelsPath = {}
labelsPath[1] = '/afs/ee.cooper.edu/courses/ece412/eventDetectionData/singlesounds/annotation1'
labelsPath[2] = '/afs/ee.cooper.edu/courses/ece412/eventDetectionData/singlesounds/annotation2'
