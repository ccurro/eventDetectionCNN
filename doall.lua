dofile('env.lua')
dofile('loadData.lua')
dofile(model)

-- batch, targets = getTrainSample()
-- x = mdl:forward(batch)

-- print(x:size())

for i = 1,nEpochs do
	dofile('train.lua')
	dofile('val.lua')
end