-- Implementation of fasttext(https://arxiv.org/abs/1607.01759) using Torch
-- Author: Junwei Pan, Yahoo Inc.
-- Date: Aug 2, 2016

require("sys")
require("nn")

local FastText = torch.class("FastText")

function FastText:__init(config)
    self.tensortype = torch.getdefaulttensortype()
    self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
    self.minfreq = config.minfreq -- minimal frequence of words to build into bocanulary
    self.dim = config.dim -- dimensions of word embeddings
    self.criterion = nn.BCECriterion() -- logistic loss
    self.n_classes = 0 -- number of classification classes
    self.labels = torch.zeros(self.n_classes)
    self.lr = config.lr -- learning rate, decayed each epoch
    self.decay = config.decay -- the flag of whether to decay the learning rate or not
    self.min_lr = config.min_lr -- minimum of the learning rate
    self.vocab = {} -- vocabulary
    self.index2word = {} -- mapping: index -> word
    self.word2index = {} -- mapping: word -> index
    self.title = config.title -- whether to use the title as features
    self.description = config.description -- whether to use the description as features
    self.n_gram = config.n_gram
    self.lst_tensor_word_idx = {}
    self.lst_labels = {}
end

-- Build vocab frequency, word2index, and index2word from input file
function FastText:build_vocab(corpus)
    print("Building vocabulary...")
    local start = sys.clock()
    local f = io.open(corpus, "r")
    local n_line = 0
    self.uniq_labels = {}
    for line in f:lines() do
	t = self:ParseCSVLine(line)
	label = t[1]
	if self.uniq_labels[label] == nil then self.uniq_labels[label] = 1 end
        for _, word in ipairs(t[2]) do
	    if self.vocab[word] == nil then
	        self.vocab[word] = 1	 
            else
	        self.vocab[word] = self.vocab[word] + 1
	    end
        end
        for _, word in ipairs(t[3]) do
	    if self.vocab[word] == nil then
	        self.vocab[word] = 1	 
            else
	        self.vocab[word] = self.vocab[word] + 1
	    end
        end
        n_line = n_line + 1
    end
    f:close()
    self.n_classes = 0
    for _ in pairs(self.uniq_labels) do self.n_classes = self.n_classes + 1 end
    -- Delete words that do not meet the minfreq threshold and create word indices
    for word, count in pairs(self.vocab) do
    	if count >= self.minfreq then
     	    self.index2word[#self.index2word + 1] = word
            self.word2index[word] = #self.index2word	    
    	else
	    self.vocab[word] = nil
        end
    end
    self.vocab_size = #self.index2word
    print(string.format("%d lines processed in %.2f seconds.", n_line, sys.clock() - start))
    print(string.format("Vocab size after eliminating words occuring less than %d times: %d", self.minfreq, self.vocab_size))
    self.word_vecs = nn.LookupTable(self.vocab_size, self.dim) -- word embeddings
    self.mean_word = nn.Sequential()
    self.mean_word:add(self.word_vecs)
    self.mean_word:add(nn.Mean(1, self.dim)) -- mean of all words in the sentence
    self.mean_word:add(nn.Reshape(1, self.dim))
    self.mean_word:reset(0.25); -- rescale N(0,1)
    self.fasttext = nn.Sequential()
    self.fasttext:add(self.mean_word)
    self.fasttext:add(nn.Linear(self.dim, self.n_classes))
    self.fasttext:add(nn.Sigmoid())
    self.decay_delta = (self.min_lr - self.lr) / n_line -- decay learning rate
end

-- Train on sentences that are streamed from the hard drive
-- Check train_mem function to train from memory (after pre-loading data into tensor)
function FastText:streaming(corpus, mode)
    if mode == "train" then
	print("Training....")
    elseif mode == "test" then
	print("Testing....")
    end

    local start = sys.clock()
    local c = 0
    local n_correct = 0.0
    f = io.open(corpus, "r")
    for line in f:lines() do
        t = self:ParseCSVLine(line)
	-- set up the label
	class = tonumber(t[1])
	self.labels = torch.zeros(self.n_classes)
	self.labels[class] = 1
	-- set up all indexs of words in the text(either title or description or both)
	t_word_idx = {}
	idx = 0
	if self.title == 1 then
	    for _, word in ipairs(t[2]) do
		word_idx = self.word2index[word]
		if word_idx ~= nil then 
		    idx = idx + 1
		    t_word_idx[idx] = word_idx
		end
	    end
	end
	if self.description == 1 then
	    for _, word in ipairs(t[3]) do
		word_idx = self.word2index[word]
		if word_idx ~= nil then 
		    idx = idx + 1
		    t_word_idx[idx] = word_idx
		end
	    end
	end
	tensor_word_idx = torch.IntTensor(#t_word_idx)
	for idx1 = 1, #t_word_idx do
	    tensor_word_idx[idx1] = t_word_idx[idx1]
	end
	if mode == "train" then
            self:train_one_sentence(tensor_word_idx, self.labels)
	elseif mode == "test" then
	    t_score = self:predict(tensor_word_idx)
	    flag_correct = self:evaluate(t_score, class)
	    n_correct = n_correct + flag_correct
	end
        c = c + 1
	if mode == "train" then
            if self.decay == 1 then self.lr = math.max(self.min_lr, self.lr + self.decay_delta) end
            if c % 10000 == 0 then
	        print(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
	    end
	elseif mode == "test" then
            if c % 10000 == 0 then
	        print(string.format("%d words processed in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
	    end
	end
    end
    if mode == "test" then
        print(string.format("Accuracy: %.4f, n_correct: %d, total_count: %d", n_correct / c, n_correct, c))
    end
end

-- Row-normalize a matrix
function FastText:normalize(m)
    m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
    	m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function FastText:get_sim_words(w, k)
    if self.word_vecs_norm == nil then
        self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    end
    if type(w) == "string" then
        if self.word2index[w] == nil then
	   print("'"..w.."' does not exist in vocabulary.")
	   return nil
	else
            w = self.word_vecs_norm[self.word2index[w]]
	end
    end
    local sim = torch.mv(self.word_vecs_norm, w)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {self.index2word[idx[i]], -sim[i]}
    end
    return r
end

-- print similar words
function FastText:print_sim_words(words, k)
    for i = 1, #words do
    	r = self:get_sim_words(words[i], k)
	if r ~= nil then
   	    print("-------"..words[i].."-------")
	    for j = 1, k do
	        print(string.format("%s, %.4f", r[j][1], r[j][2]))
	    end
	end
    end
end

-- print similar words in an interactive way
function FastText:print_sim_words_interactive(k)
    print("Please input the words")
    while true do
        local line = io.read()
        if line == nil then break end
        if self.word2index[line] == nil then 
            print(string.format("%s is not in the vocabulary", line))
        else
            r = self:get_sim_words(line, k)
            for j = 1, k do
                print(string.format("%s, %.4f", r[j][1], r[j][2]))
            end
        end
    end
end

-- Concat the contents of the parameter list,
-- separated by the string delimiter (just like in perl)
-- example: strjoin(", ", {"Anna", "Bob", "Charlie", "Dolores"})
function FastText:join_str(list, delimiter)
    local len = table.getn(list)
    if len == 0 then 
        return "" 
    end
    local string = list[1]
    for i = 2, len do 
        string = string .. delimiter .. list[i] 
    end
    return string
end

-- Split text into a list consisting of the strings in text,
-- separated by strings matching delimiter (which may be a pattern). 
-- example: strsplit(",%s*", "Anna, Bob, Charlie,Dolores")
function FastText:split_str(text, delimiter)
    local list = {}
    local pos = 1
    if string.find("", delimiter, 1) then -- this would result in endless loops
        error("delimiter matches empty string!")
    end
    while 1 do
        local first, last = string.find(text, delimiter, pos)
        if first then -- found?
            table.insert(list, string.sub(text, pos, first-1))
            pos = last+1
        else
            table.insert(list, string.sub(text, pos))
            break
        end
    end
    return list
end

function FastText:delete_punc(string)
    res = self:join_str(self:split_str(string, "%p"), "")
    return res
end

function FastText:add_bigram(t)
    for idx = 1, #t - 1 do
        word_current = t[idx]
        word_next = t[idx + 1]
        t[#t + 1] = word_current .. " " .. word_next
    end
    return t
end

function FastText:add_trigram(t)
    for idx = 1, #t - 2 do
        word_current = t[idx]
        word_next = t[idx + 1]
	word_next_next = t[idx + 2]
        t[#t + 1] = word_current .. " " .. word_next .. " " .. word_next_next
    end
    return t
end

-- split each line to get a table where:
-- t[1] is the class
-- t[2] and t[3] is a table of words for title and descriptions respectively.
function FastText:ParseCSVLine(line, sep) 
    local res = {}
    local pos = 1
    sep = sep or ','
    while true do 
        local c = string.sub(line,pos,pos)
        if (c == "") then break end
        if (c == '"') then
	    -- quoted value (ignore separator within)
	    local txt = ""
	    repeat
	        local startp,endp = string.find(line,'^%b""',pos)
	        txt = txt..string.sub(line,startp+1,endp-1)
	        pos = endp + 1
	        c = string.sub(line,pos,pos) 
	        if (c == '"') then txt = txt..'"' end 
	        -- check first char AFTER quoted string, if it is another
	        -- quoted string without separator, then append it
	        -- this is the way to "escape" the quote char in a quote. example:
	        --   value1,"blub""blip""boing",value3  will result in blub"blip"boing  for the middle
	    until (c ~= '"')
	    table.insert(res,txt)
	    assert(c == sep or c == "")
	    pos = pos + 1
        else
	    -- no quotes used, just look for the first separator
	    local startp,endp = string.find(line,sep,pos)
	    if (startp) then 
	        table.insert(res,string.sub(line,pos,startp-1))
	        pos = endp + 1
	    else
	        -- no separator found -> use rest of string and terminate
	        table.insert(res,string.sub(line,pos))
	        break
	    end 
        end
     end
    assert(#res == 3)
    res[2] = self:split_str(self:delete_punc(res[2]), " ")
    res[3] = self:split_str(self:delete_punc(res[3]), " ")
    assert(self.n_gram == 1 or self.n_gram == 2 or self.n_gram == 3)
    if self.n_gram >= 2 then
        res[2] = self:add_bigram(res[2])
        res[3] = self:add_bigram(res[3])
    end
    if self.n_gram == 3 then
        res[2] = self:add_trigram(res[2])
        res[3] = self:add_trigram(res[3])
    end
   return res
end

-- Train on word context pairs
function FastText:train_one_sentence(tensor_word_idx, labels)
    if tensor_word_idx:nDimension() > 0 then
        local p = self.fasttext:forward(tensor_word_idx)
        local loss = self.criterion:forward(p, labels)
        local dl_dp = self.criterion:backward(p, labels)
        self.fasttext:zeroGradParameters()
        self.fasttext:backward(tensor_word_idx, dl_dp)
        self.fasttext:updateParameters(self.lr)
    end
end

-- Test on test data
function FastText:predict(tensor_word_idx)
    if tensor_word_idx:dim() <= 0 then tensor_word_idx = torch.IntTensor(10); tensor_word_idx:fill(1) end
    local p = self.fasttext:forward(tensor_word_idx)
    return p
end

function FastText:evaluate(t_score, class)
    max = -1
    index = 0
    t_score = t_score[1]
    for idx = 1, self.n_classes do
	score = t_score[idx]
	if score > max then
	    max = score
	    index = idx
	end
    end
    if index == class then return 1 else return 0 end
end

function FastText:preload_data(corpus)
    print("Loading the data into the memory")
    local start = sys.clock()
    local c = 0
    self.lst_labels = {}
    self.lst_tensor_word_idx = {}
    f = io.open(corpus, "r")
    for line in f:lines() do
        c = c + 1
        t = self:ParseCSVLine(line)
	class = t[1]
	labels = torch.zeros(self.n_classes)
	labels[class] = 1
	self.lst_labels[c] = labels
	t_word_idx = {}
	idx = 0
	if self.title == 1 then
	    for _, word in ipairs(t[2]) do
		word_idx = self.word2index[word]
		if word_idx ~= nil then 
		    idx = idx + 1
		    t_word_idx[idx] = word_idx
		end
	    end
	end
	if self.description == 1 then
	    for _, word in ipairs(t[3]) do
		word_idx = self.word2index[word]
		if word_idx ~= nil then 
		    idx = idx + 1
		    t_word_idx[idx] = word_idx
		end
	    end
	end
        tensor_word_idx = torch.IntTensor(#t_word_idx)
        for idx1 = 1, #t_word_idx do
            tensor_word_idx[idx1] = t_word_idx[idx1]
        end
        self.lst_tensor_word_idx[c] = tensor_word_idx
    end
    print(string.format("%d lines loaded in the memory in %.2f seconds", c, sys.clock() - start))
end

-- train from memory. this is needed to speed up GPU training
function FastText:train_mem()
    local start = sys.clock()
    for i = 1, #self.lst_labels do
        self:train_one_sentence(self.lst_tensor_word_idx[i], self.lst_labels[i])
	if self.decay == 1 then self.lr = math.max(self.min_lr, self.lr + self.decay_delta) end
        if i % 10000 == 0 then
            print(string.format("%d sentences trained in %.2f seconds. Learning rate: %.4f", i, sys.clock() - start, self.lr))
	end
    end    
end

-- train the model using config parameters
function FastText:train_model(corpus)
    if self.stream == 1 then
        self:streaming(corpus, "train")
    else
        self:preload_data(corpus)
	self:train_mem()
    end
end

-- test the model using config parameters
function FastText:test_model(corpus)
    self:streaming(corpus, "test")
end

-- save model to disc
function FastText:save_model(path)
    torch.save(path, self)
end

-- save vectors for each word
function FastText:save_vector(path)
    if self.word_vecs_norm == nil then
        self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    end
    t = {}
    for i = 1, self.vocab_size do
        word = self.index2word[i]
        v = self.word_vecs_norm[i]
        t[word] = v
    end
    torch.save(path, t, 'ascii')
end
