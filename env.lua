require 'nn'
require 'cutorch'
require 'cunn'
-- require 'rnn'
require 'optim'
require 'audio'
require 'signal'
require 'gnuplot'
require 'torchx'
require 'image'
require('./sampleAq')

soundsPath  = '/afs/ee.cooper.edu/courses/ece412/eventDetectionData/singlesounds/sounds'

labelsPath = {}
labelsPath[1] = '/afs/ee.cooper.edu/courses/ece412/eventDetectionData/singlesounds/annotation1'
labelsPath[2] = '/afs/ee.cooper.edu/courses/ece412/eventDetectionData/singlesounds/annotation2'

fs = 44.1e3
superFrameSize = 20e-3*fs

nModel = os.time()

logFile = io.open(string.format('logs/model%d.err',nModel),'a')
s = torch.initialSeed()
logFile:write(string.format('Seed: %d\n',s))
logFile:close()

criterion = nn.ClassNLLCriterion()
criterion:cuda()

nEpochs = 1e3
epochSize = 400
epoch = 1
batchSize = 32
cvError = torch.Tensor(nEpochs)

optimState = {
    learningRate = 1e-4,
    weightDecay = 0, 
    momentum = 0.9,
    learningRateDecay = 0
}

optimMethod = optim.nag

model = 'model.lua'