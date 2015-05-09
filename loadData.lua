dofile('env.lua')

-- huge refactoring neeeded

print('==> Indexing Audio files')

local soundsFiles = paths.indexdir(soundsPath, 'wav')

local labelsFiles = {}
labelsFiles[1] = paths.indexdir(labelsPath[1], 'txt') 
labelsFiles[2] = paths.indexdir(labelsPath[2], 'txt')

local labels = {
	alert = 1,
	clearthroat = 2,
	cough = 3,
	doorslam = 4,
	drawer = 5,
	keyboard = 6,
	keys = 7,
	knock = 8,
	laughter = 9,
	mouse = 10,
	pageturn = 11,
	pendrop = 12,
	phone = 13,
	printer = 14,
	speech = 15,
	switch = 16,
	none = 17
}

local function readLabel(filename)
	-- filename is a string containing the full path to the wav file
	local name = string.match(filename, "([^/]+)$") -- strip to just filename
	local name, id = string.match(string.gsub(name,'.wav$',''),'(.-)(%d+)') -- cutoff 
	local label = labels[name]
	return label, tonumber(id)
end

local function readSetTimes(filename)
	local file = io.open(filename)
	local str = file:read()
	local setTimes = {}
	for i in string.gmatch(str, "%S+") do
		table.insert(setTimes,tonumber(i))
	end
	return setTimes
end	

local dataset = {}


print('==> Loading Audio Files')
for i = 1,soundsFiles:size() do
	local filename = soundsFiles:filename(i)
	local p = {} -- sample, label pair
	p['audio'] = audio.load(filename):select(1,1)
	p['label'], p['id'] = readLabel(filename)
	local setTimes = {}
	setTimes[1] = readSetTimes(labelsFiles[1]:filename(i))
	setTimes[2] = readSetTimes(labelsFiles[2]:filename(i))	
	p['on']  = (setTimes[1][1] + setTimes[2][1])*0.5
	p['off'] = (setTimes[1][2] + setTimes[2][2])*0.5
	table.insert(dataset,p)

	-- p['audio'] = audio.load(filename):select(1,2)
	-- table.insert(dataset,p)
	if i % 5 == 0 then 
		xlua.progress(i,soundsFiles:size())
	end
end

local function shuffle( t )
    local rand = math.random
    assert( t, "shuffle() expected a table, got nil" )
    local iterations = #t
    local j
    
    for i = iterations, 2, -1 do
        j = rand(i)
        t[i], t[j] = t[j], t[i]
    end
end

print('==> Shuffling Dataset')
shuffle(dataset)

local trainSet = {}
local valSet = {}

local trainingSplit = .8

print('==> Splitting Dataset')
for i = 1,#dataset do
	if i < torch.floor(trainingSplit*#dataset) then
		table.insert(trainSet,dataset[i])
	else
		table.insert(valSet,dataset[i])
	end
end

-- refactoring needed

local function getSuperFrames(example)
	-- example is item of dataSet containing 
	-- the following fields: audio, label, on, off
    local superFrames = {}

	local onSet  = example['on']*fs
    local offSet = example['off']*fs
    for i = 1,torch.floor(example['audio']:numel()/superFrameSize) do
        local frame = {}
        frame['label'] = example['label']
        frame['audio'] = example['audio'][{{1 + (i-1)*superFrameSize,i*superFrameSize}}]
        if (onSet > 1 + (i-1)*superFrameSize) or (offSet < i*superFrameSize) then
            frame['event'] = false
        else
            frame['event'] = true
        end
        if onSet > (1 + (i-1)*superFrameSize) and onSet < (1 + i*superFrameSize) then
            frame['on'] = true   
            frame['onSet'] = onSet - (1 + (i-1)*superFrameSize)
        else
            frame['on'] = false
        end
        if offSet > (1 + (i-1)*superFrameSize) and offSet < (1 + i*superFrameSize) then
            frame['off'] = true   
            frame['offSet'] = offSet - (1 + i*superFrameSize)
        else
            frame['off'] = false
        end
        superFrames[i] = frame
    end
    return superFrames
end

local trainSuperFrames = {}

for i = 1,#trainSet do
    for k,v in ipairs(getSuperFrames(trainSet[i])) do
        table.insert(trainSuperFrames, v)
    end
end

local valSuperFrames = {}

for i = 1,#valSet do
    for k,v in ipairs(getSuperFrames(valSet[i])) do
        table.insert(valSuperFrames, v)
    end
end

trainClassSuperFrames = {}
for i = 1,16 do
    table.insert(trainClassSuperFrames,{})
end

for k,v in ipairs(trainSuperFrames) do
    local frame = {}
    if v['event'] then
        -- frame['label'] = v['label']
        frame['audio'] = v['audio']
        table.insert(trainClassSuperFrames[v['label']],frame)
    else
        frame['label'] = 17 -- none
    end
end

shuffle(trainClassSuperFrames)

trainOnRegressSuperFrames = {}

for k,v in ipairs(trainSuperFrames) do
    local frame = {}
    if v['on'] then
        frame['audio'] = v['audio']
        frame['onSet'] = v['onSet']
        table.insert(trainOnRegressSuperFrames,frame)
    end
end

shuffle(trainOnRegressSuperFrames)

trainOffRegressSuperFrames = {}

for k,v in ipairs(trainSuperFrames) do
    local frame = {}
    if v['off'] then
        frame['audio'] = v['audio']
        frame['offSet'] = v['offSet']
        table.insert(trainOffRegressSuperFrames,frame) 
    end
end

shuffle(trainOffRegressSuperFrames)

valClassSuperFrames = {}

for k,v in ipairs(valSuperFrames) do
    local frame = {}
    if v['event'] then
        frame['label'] = v['label']
        frame['audio'] = v['audio']
        table.insert(valClassSuperFrames,frame)
    else
        frame['label'] = 17 -- none
    end
end

shuffle(valClassSuperFrames)

valOnRegressSuperFrames = {}

for k,v in ipairs(valSuperFrames) do
    local frame = {}
    if v['on'] then
        frame['audio'] = v['audio']
        frame['onSet'] = v['onSet']
        table.insert(valOnRegressSuperFrames,frame)
    end
end

shuffle(valOnRegressSuperFrames)

valOffRegressSuperFrames = {}

for k,v in ipairs(valSuperFrames) do
    local frame = {}
    if v['off'] then
        frame['audio'] = v['audio']
        frame['offSet'] = v['offSet']
        table.insert(valOffRegressSuperFrames,frame) 
    end
end

shuffle(valOffRegressSuperFrames)
