----------------------------------------------------------------------
print '==> train.lua'

mdl:training()

parameters,gradParameters = mdl:getParameters()

print '==> defining training procedure'
function train()
   local time = sys.clock()
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

local epochError = 0 
for t = 1,epochSize do

   local batch, targets = getTrainSample()
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end
         gradParameters:zero()
         local f = 0;
         local oHat = mdl:forward(batch)
         f = f + criterion:forward(oHat,targets)
         mdl:backward(batch,criterion:backward(oHat,targets)) --problem line
	      epochError = epochError + f
            print('# of Examples:',t*batchSize,'Error:',f)
            return f,gradParameters
      end
      optimMethod(feval, parameters, optimState)
      collectgarbage()
   end
   -- time taken
time = sys.clock() - time
print("<trainer> time for 1 Epoch = " .. (time) .. 's')
epochError = epochError/epochSize
local errStr = string.format('Epoch: %g, Epoch Error: %g, Learning Rate: %g, Decay: %g',epoch,epochError,optimState.learningRate,optimState.weightDecay)
print(errStr)
-- local errFile = io.open(logFileName,'a')
local logFile = io.open(string.format('logs/model%d.err',nModel),'a')
-- logFile:open()
logFile:write(errStr)
logFile:close()
end

train()
