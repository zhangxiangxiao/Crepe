--[[
Model Program for Crepe
By Xiang Zhang @ New York University
--]]

-- Prerequisite
require("nn")

-- The class
local Model = torch.class("Model")

function Model:__init(config)
   -- Create a sequential for self
   if config.file then
      self.sequential = Model:makeCleanSequential(torch.load(config.file))
   else
      self.sequential = Model:createSequential(config)
   end
   self.p = config.p or 0.5
   self.tensortype = torch.getdefaulttensortype()
end

-- Get the parameters of the model
function Model:getParameters()
   return self.sequential:getParameters()
end

-- Forward propagation
function Model:forward(input)
   self.output = self.sequential:forward(input)
   return self.output
end

-- Backward propagation
function Model:backward(input, gradOutput)
   self.gradInput = self.sequential:backward(input, gradOutput)
   return self.gradInput
end

-- Randomize the model to random parameters
function Model:randomize(sigma)
   local w,dw = self:getParameters()
   w:normal():mul(sigma or 1)
end

-- Enable Dropouts
function Model:enableDropouts()
   self.sequential = self:changeSequentialDropouts(self.sequential, self.p)
end

-- Disable Dropouts
function Model:disableDropouts()
   self.sequential = self:changeSequentialDropouts(self.sequential,0)
end

-- Switch to a different data mode
function Model:type(tensortype)
   if tensortype ~= nil then
      self.sequential = self:makeCleanSequential(self.sequential)
      self.sequential:type(tensortype)
      self.tensortype = tensortype
   end
   return self.tensortype
end

-- Switch to cuda
function Model:cuda()
   self:type("torch.CudaTensor")
end

-- Switch to double
function Model:double()
   self:type("torch.DoubleTensor")
end

-- Switch to float
function Model:float()
   self:type("torch.FloatTensor")
end

-- Change dropouts
function Model:changeSequentialDropouts(model,p)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
	   m.p = p
      end
   end
   return model
end

-- Create a sequential model using configurations
function Model:createSequential(model)
   local new = nn.Sequential()
   for i,m in ipairs(model) do
      new:add(Model:createModule(m))
   end
   return new
end

-- Clear the module out of gradient data and input/output
function Model:clearSequential(model)
   for i,m in ipairs(model.modules) do
      if m.output then m.output = torch.Tensor() end
      if m.gradInput then m.gradInput = torch.Tensor() end
      if m.gradWeight then m.gradWeight = torch.Tensor() end
      if m.gradBias then m.gradBias = torch.Tensor() end
   end
   return model
end

-- Make a clean sequential model
function Model:makeCleanSequential(model)
   local new = nn.Sequential()
   for i = 1,#model.modules do
      local m = Model:makeCleanModule(model.modules[i])
      if m then
	 new:add(m)
      end
   end
   return new
end

-- Create a module using configurations
function Model:createModule(m)
   if m.module == "nn.Reshape" then
      return Model:createReshape(m)
   elseif m.module == "nn.Linear" then
      return Model:createLinear(m)
   elseif m.module == "nn.Threshold" then
      return Model:createThreshold(m)
   elseif m.module == "nn.TemporalConvolution" then
      return Model:createTemporalConvolution(m)
   elseif m.module == "nn.TemporalMaxPooling" then
      return Model:createTemporalMaxPooling(m)
   elseif m.module == "nn.Dropout" then
      return Model:createDropout(m)
   elseif m.module == "nn.LogSoftMax" then
      return Model:createLogSoftMax(m)
   else
      error("Unrecognized module for creation: "..tostring(m.module))
   end
end

-- Make a clean module
function Model:makeCleanModule(m)
   if torch.typename(m) == "nn.TemporalConvolution" then
	 return Model:toTemporalConvolution(m)
   elseif torch.typename(m) == "nn.Threshold" then
      return Model:newThreshold()
   elseif torch.typename(m) == "nn.TemporalMaxPooling" then
      return Model:toTemporalMaxPooling(m)
   elseif torch.typename(m) == "nn.Reshape" then
      return Model:toReshape(m)
   elseif torch.typename(m) == "nn.Linear" then
      return Model:toLinear(m)
   elseif torch.typename(m) == "nn.LogSoftMax" then
      return Model:newLogSoftMax(m)
   elseif torch.typename(m) == "nn.Dropout" then
      return Model:toDropout(m)
   else
      error("Module unrecognized")
   end
end


-- Create a new reshape model
function Model:createReshape(m)
   return nn.Reshape(m.size)
end

-- Create a new linear model
function Model:createLinear(m)
   return nn.Linear(m.inputSize, m.outputSize)
end

-- Create a new threshold model
function Model:createThreshold(m)
   return nn.Threshold()
end

-- Create a new Spatial Convolution model
function Model:createTemporalConvolution(m)
   return nn.TemporalConvolution(m.inputFrameSize, m.outputFrameSize, m.kW, m.dW)
end

-- Create a new spatial max pooling model
function Model:createTemporalMaxPooling(m)
   return nn.TemporalMaxPooling(m.kW, m.dW, m.dH)
end

-- Create a new dropout module
function Model:createDropout(m)
   return nn.Dropout(m.p)
end

-- Create new logsoftmax module
function Model:createLogSoftMax(m)
   return nn.LogSoftMax()
end

-- Create a new threshold
function Model:newThreshold()
   return nn.Threshold()
end

-- Convert to a new max pooling
function Model:toTemporalMaxPooling(m)
   return nn.TemporalMaxPooling(m.kW, m.dW)
end

-- Convert to a new reshape
function Model:toReshape(m)
   return nn.Reshape(m.size)
end

-- Convert to a new dropout
function Model:toDropout(m)
   return nn.Dropout(m.p)
end

-- Convert to a new linear module
function Model:toLinear(m)
   local new = nn.Linear(m.weight:size(2),m.weight:size(1))
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end

-- Create a new LogSoftMax
function Model:newLogSoftMax()
   return nn.LogSoftMax()
end

-- Convert a convolution module to standard
function Model:toTemporalConvolution(m)
   local new = nn.TemporalConvolution(m.inputFrameSize, m.outputFrameSize, m.kW, m.dW)
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end
