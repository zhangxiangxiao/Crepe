--[[
Trainer for Crepe
By Xiang Zhang @ New York University
--]]

require("sys")

local Train = torch.class("Train")

-- Initialization of the trainer class
-- data: the data object
-- model: the model object
-- loss: the loss object
-- config: (optional) the configuration table
--      .rates: (optional) the table of learning rates, indexed by the number of epoches
--      .epoch: (optional) current epoch
function Train:__init(data,model,loss,config)
   -- Store the objects
   self.data = data
   self.model = model
   self.loss = loss

   -- Store the configurations and states
   local config = config or {}
   self.rates = config.rates or {1e-3}
   self.epoch = config.epoch or 1

   -- Get the parameters and gradients
   self.params, self.grads = self.model:getParameters()
   self.old_grads = self.grads:clone():zero()

   -- Make the loss correct type
   self.loss:type(self.model:type())

   -- Find the current rate
   local max_epoch = 1
   self.rate = self.rates[1]
   for i,v in pairs(self.rates) do
      if i <= self.epoch and i > max_epoch then
	 max_epoch = i
	 self.rate = v
      end
   end

   -- Timing table
   self.time = {}

   -- Store the configurations
   self.momentum = config.momentum or 0
   self.decay = config.decay or 0
   self.normalize = config.normalize
   self.recapture = config.recapture
end

-- Run for a number of steps
-- epoches: number of epoches
-- logfunc: (optional) a function to execute after each step.
function Train:run(epoches,logfunc)
   -- Recapture the weights
   if self.recapture then
      self.params,self.grads = nil,nil
      collectgarbage()
      self.params,self.grads = self.model:getParameters()
      collectgarbage()
   end
   -- The loop
   for i = 1,epoches do
      self:batchStep()
      if logfunc then logfunc(self,i) end
   end
end

-- Run for one batch step
function Train:batchStep()
   self.clock = sys.clock()
   -- Get a batch of data
   self.batch_untyped,self.labels_untyped = self.data:getBatch(self.batch_untyped,self.labels_untyped)
   -- Make the data to correct type
   self.batch = self.batch or self.batch_untyped:transpose(2, 3):contiguous():type(self.model:type())
   self.labels = self.labels or self.labels_untyped:type(self.model:type())
   self.batch:copy(self.batch_untyped:transpose(2, 3):contiguous())
   self.labels:copy(self.labels_untyped)
   -- Record time
   if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
   self.time.data = sys.clock() - self.clock

   self.clock = sys.clock()
   -- Forward propagation
   self.output = self.model:forward(self.batch)
   self.objective = self.loss:forward(self.output,self.labels)
   if type(self.objective) ~= "number" then self.objective = self.objective[1] end
   self.max, self.decision = self.output:double():max(2)
   self.max = self.max:squeeze():double()
   self.mask = self.labels:double():gt(0):double()
   self.decision = self.decision:squeeze():double()
   if self.mask:sum() > 0 then
      self.error = torch.ne(self.decision,self.labels:double()):double():cmul(self.mask):sum()/self.mask:sum()
   else
      self.error = 1
   end
   -- Record time
   if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
   self.time.forward = sys.clock() - self.clock

   self.clock = sys.clock()
   -- Backward propagation   
   self.grads:zero()
   self.gradOutput = self.loss:backward(self.output,self.labels)
   self.gradBatch = self.model:backward(self.batch,self.gradOutput)
   -- Record time
   if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
   self.time.backward = sys.clock() - self.clock

   self.clock = sys.clock()
   -- Update the step
   self.old_grads:mul(self.momentum):add(self.grads:mul(-self.rate))
   self.params:mul(1-self.rate*self.decay):add(self.old_grads)
   if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
   self.time.update = sys.clock() - self.clock

   -- Increment on the epoch
   self.epoch = self.epoch + 1
   -- Change the learning rate
   self.rate = self.rates[self.epoch] or self.rate
end
