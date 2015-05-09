local labels = {
   'alert',
   'clearthroat',
   'cough',
   'doorslam',
   'drawer',
   'keyboard',
   'keys',
   'knock',
   'laughter',
   'mouse',
   'pageturn',
   'pendrop',
   'phone',
   'printer',
   'speech',
   'switch'
   -- 'none'
}

mdl:evaluate()
local conf = optim.ConfusionMatrix(labels)
conf:zero()
local valError = 0
local valSize = torch.floor(#valClassSuperFrames/batchSize)
local time = sys.clock()
for i=1,valSize do
   local batch, targets = getValSample(i)
   local oHat = mdl:forward(batch)
   conf:batchAdd(oHat,targets)
   valError = valError + criterion:forward(oHat,targets)
   if i == valSize then
      cvError[epoch] = valError/valSize
      local errStr = string.format(' Cross Val Error: %g\n',valError/valSize)
      print(errStr)
      local logFile = io.open(string.format('logs/model%d.err',nModel),'a')
      -- logFile:open()
      logFile:write(errStr)
      logFile:close()
   end
end
-- print(conf)
image.save('conf.png',conf:render())
time = sys.clock() - time
print("<validation> time for CrosVal = " .. (time) .. 's')
epoch = epoch + 1
torch.seed()
collectgarbage()