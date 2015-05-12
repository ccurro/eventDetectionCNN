-- function getTrainSample()
--     -- local batch = torch.Tensor(batchSize,33,26)
--     local batch = torch.Tensor(batchSize,882,1)
--     local targets = torch.Tensor(batchSize)
--     for class = 1,16 do
--         local sampleIndex = torch.randperm(#trainClassSuperFrames[class]):narrow(1,1,batchSize/16)
--         local count = 1     
--         for i = (1 + (class-1)*batchSize/16),((class)*batchSize/16) do
--             local ind   = sampleIndex[count]
--             local sound = trainClassSuperFrames[class][ind]['audio']
--             batch[i] = sound:view(882,1)
--             -- batch[i]    = audio.spectrogram(sound:view(1,882),64,'hamming',32)
--             targets[i]  = class
--             count = count + 1
--         end
--     end
--     return batch:cuda(), targets:cuda()
-- end

function getValSample(n)
    local sampleIndex = torch.range(1,batchSize):add(batchSize*(n-1))
    local batch = torch.Tensor(batchSize,33,26)
    -- local batch = torch.Tensor(batchSize,882,1)
    local targets = torch.Tensor(batchSize)
    for i = 1,sampleIndex:size(1) do
        local ind   = sampleIndex[i]
        local sound = valClassSuperFrames[ind]['audio']
        -- batch[i] = sound:view(882,1)
        batch[i]    = audio.spectrogram(sound:view(1,882),64,'hann',32)
        targets[i]  = valClassSuperFrames[ind]['label']
    end
    return batch:cuda(), targets:cuda()
end

function getTrainSample()
    local sampleIndex = torch.randperm(#trainClassSuperFrames):narrow(1,1,batchSize)
    local batch   = torch.Tensor(batchSize,33,26) -- size based on window size and stft params
    local targets = torch.Tensor(batchSize)
    for i = 1,batchSize do
        local ind  = sampleIndex[i]
        local sound = trainClassSuperFrames[ind]['audio']
        batch[i] = audio.spectrogram(sound:view(1,882),64,'hann',32)
        targets[i] = trainClassSuperFrames[ind]['label']
    end
    return batch:cuda(), targets:cuda()
end