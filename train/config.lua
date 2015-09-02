--[[
Configuration for Crepe Training Program
By Xiang Zhang @ New York University
--]]

require("nn")

-- The namespace
config = {}

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

-- Training data
config.train_data = {}
config.train_data.file = paths.concat(paths.cwd(), "../data/train.t7b")
config.train_data.alphabet = alphabet
config.train_data.length = 1014
config.train_data.batch_size = 128

-- Validation data
config.val_data = {}
config.val_data.file =  paths.concat(paths.cwd(), "../data/test.t7b")
config.val_data.alphabet = alphabet
config.val_data.length = 1014
config.val_data.batch_size = 128

-- The model
config.model = {}
-- #alphabet x 1014
config.model[1] = {module = "nn.TemporalConvolution", inputFrameSize = #alphabet, outputFrameSize = 256, kW = 7}
config.model[2] = {module = "nn.Threshold"}
config.model[3] = {module = "nn.TemporalMaxPooling", kW = 3, dW = 3}
-- 336 x 256
config.model[4] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 7}
config.model[5] = {module = "nn.Threshold"}
config.model[6] = {module = "nn.TemporalMaxPooling", kW = 3, dW = 3}
-- 110 x 256
config.model[7] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[8] = {module = "nn.Threshold"}
-- 108 x 256
config.model[9] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[10] = {module = "nn.Threshold"}
-- 106 x 256
config.model[11] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[12] = {module = "nn.Threshold"}
-- 104 x 256
config.model[13] = {module = "nn.TemporalConvolution", inputFrameSize = 256, outputFrameSize = 256, kW = 3}
config.model[14] = {module = "nn.Threshold"}
config.model[15] = {module = "nn.TemporalMaxPooling", kW = 3, dW = 3}
-- 34 x 256
config.model[16] = {module = "nn.Reshape", size = 8704}
-- 8704
config.model[17] = {module = "nn.Linear", inputSize = 8704, outputSize = 1024}
config.model[18] = {module = "nn.Threshold"}
config.model[19] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[20] = {module = "nn.Linear", inputSize = 1024, outputSize = 1024}
config.model[21] = {module = "nn.Threshold"}
config.model[22] = {module = "nn.Dropout", p = 0.5}
-- 1024
config.model[23] = {module = "nn.Linear", inputSize = 1024, outputSize = 14}
config.model[24] = {module = "nn.LogSoftMax"}

-- The loss
config.loss = nn.ClassNLLCriterion

-- The trainer
config.train = {}
local baseRate = 1e-2 * math.sqrt(config.train_data.batch_size) / math.sqrt(128)
config.train.rates = {[1] = baseRate/1,[15001] = baseRate/2,[30001] = baseRate/4,[45001] = baseRate/8,[60001] = baseRate/16,[75001] = baseRate/32,[90001]= baseRate/64,[105001] = baseRate/128,[120001] = baseRate/256,[135001] = baseRate/512,[150001] = baseRate/1024}
config.train.momentum = 0.9
config.train.decay = 1e-5

-- The tester
config.test = {}
config.test.confusion = true

-- UI settings
config.mui = {}
config.mui.width = 1200
config.mui.scale = 4
config.mui.n = 16

-- Main program
config.main = {}
config.main.type = "torch.CudaTensor"
config.main.eras = 10
config.main.epoches = 5000
config.main.randomize = 5e-2
config.main.dropout = true
config.main.save = paths.concat(paths.cwd())
config.main.details = true
config.main.device = 1
config.main.collectgarbage = 100
config.main.logtime = 5
config.main.debug = false
config.main.test = true
