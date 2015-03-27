dofile('env.lua')

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
	switch = 16
}

function readLabel(filename)
	-- filename is a string containing the full path to the wav file
	local name = string.match(filename, "([^/]+)$") -- strip to just filename
	local name, id = string.match(string.gsub(name,'.wav$',''),'(.-)(%d+)') -- cutoff 
	local label = labels[name]
	return label
end

local dataset = {}

for i = 1,soundsFiles:size() do
	local filename = soundsFiles:filename(i)
	local p = {} -- sample, label pair
	p['audio'] = audio.load(filename):select(1,1)
	p['label'] = readLabel(filename)
	table.insert(dataset,p)

	p['audio'] = audio.load(filename):select(1,2)
	table.insert(dataset,p)
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

shuffle(dataset)

trainingSet = {}
valSet = {}

local trainingSplit = .8

for i = 1,#dataset do
	if i < torch.floor(trainingSplit*#dataset) then
		table.insert(trainingSet,dataset[i])
	else
		table.insert(valSet,dataset[i])
	end
end

