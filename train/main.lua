--[[
Main Driver for Crepe
By Xiang Zhang @ New York University
]]

-- Necessary functionalities
require("nn")
require("cutorch")
require("cunn")
require("gnuplot")

-- Local requires
require("data")
require("model")
require("train")
require("test")
require("mui")

-- Configurations
dofile("config.lua")

-- Prepare random number generator
math.randomseed(os.time())
torch.manualSeed(os.time())

-- Create namespaces
main = {}

-- The main program
function main.main()
   -- Setting the device
   if config.main.device then
      cutorch.setDevice(config.main.device)
      print("Device set to "..config.main.device)
   end

   main.clock = {}
   main.clock.log = 0

   main.argparse()
   main.new()
   main.run()
end

-- Parse arguments
function main.argparse()
   local cmd = torch.CmdLine()

   -- Options
   cmd:option("-resume",0,"Resumption point in epoch. 0 means not resumption.")
   cmd:text()
   
   -- Parse the option
   local opt = cmd:parse(arg or {})
   
   -- Resumption operation
   if opt.resume > 0 then
      -- Find the main resumption file
      local files = main.findFiles(paths.concat(config.main.save,"main_"..tostring(opt.resume).."_*.t7b"))
      if #files ~= 1 then
	 error("Found "..tostring(#files).." main resumption point.")
      end
      config.main.resume = files[1]
      print("Using main resumption point "..config.main.resume)
      -- Find the model resumption file
      local files = main.findFiles(paths.concat(config.main.save,"sequential_"..tostring(opt.resume).."_*.t7b"))
      if #files ~= 1 then
	 error("Found "..tostring(#files).." model resumption point.")
      end
      config.model.file = files[1]
      print("Using model resumption point "..config.model.file)
      -- Resume the training epoch
      config.train.epoch = tonumber(opt.resume) + 1
      print("Next training epoch resumed to "..config.train.epoch)
      -- Don't do randomize
      if config.main.randomize then
	 config.main.randomize = nil
	 print("Disabled randomization for resumption")
      end
   end

   return opt
end

-- Train a new experiment
function main.new()
   -- Load the data
   print("Loading datasets...")
   main.train_data = Data(config.train_data)
   main.val_data = Data(config.val_data)

   -- Load the model
   print("Loading the model...")
   main.model = Model(config.model)
   if config.main.randomize then
      main.model:randomize(config.main.randomize)
      print("Model randomized.")
   end
   main.model:type(config.main.type)
   print("Current model type: "..main.model:type())
   collectgarbage()

   -- Initiate the trainer
   print("Loading the trainer...")
   main.train = Train(main.train_data, main.model, config.loss(), config.train)

   -- Initiate the tester
   print("Loading the tester...")
   main.test_train = Test(main.train_data, main.model, config.loss(), config.test)
   main.test_val = Test(main.val_data, main.model, config.loss(), config.test)

   -- The record structure
   main.record = {}
   if config.main.resume then
      print("Loading main record...")
      local resume = torch.load(config.main.resume)
      main.record = resume.record
      if resume.momentum then main.train.old_grads:copy(resume.momentum) end
      main.show()
   end

   -- The visualization
   main.mui = Mui{width=config.mui.width,scale=config.mui.scale,n=config.mui.n,title="Model Visualization"}
   main.draw()
   collectgarbage()
end

-- Start the training
function main.run()
   --Run for this number of era
   for i = 1,config.main.eras do
      if config.main.dropout then
	 print("Enabling dropouts")
	 main.model:enableDropouts()
      else
	 print("Disabling dropouts")
	 main.model:disableDropouts()
      end
      print("Training for era "..i)
      main.train:run(config.main.epoches, main.trainlog)

      print("Disabling dropouts")
      main.model:disableDropouts()
      print("Testing on training data for era "..i)
      main.test_train:run(main.testlog)

      if config.main.test == nil or config.main.test == true then
	 print("Disabling dropouts")
	 print("Testing on test data for era "..i)
	 main.test_val:run(main.testlog)
      end

      print("Recording on era "..i)
      main.record[#main.record+1] = {train_error = main.test_train.e,
				     train_loss = main.test_train.l,
				     val_error = main.test_val.e,
				     val_loss = main.test_val.l}
      if config.test.confusion then
	 main.record[#main.record].train_confusion = main.test_train.confusion:clone()
	 main.record[#main.record].val_confusion = main.test_val.confusion:clone()
      end
      
      print("Visualizing loss")
      main.show()
      print("Visualizing the models")
      main.draw()
      print("Saving data")
      main.save()
      collectgarbage()
   end
end

-- Final cleaning up
function main.clean()
   print("Cleaning up...")
   gnuplot.closeall()
end

-- Draw the graph
function main.show(figure_error,figure_loss)
   main.figure_error = main.figure_error or gnuplot.figure()
   main.figure_loss = main.figure_loss or gnuplot.figure()

   local figure_error = figure_error or main.figure_error
   local figure_loss = figure_loss or main.figure_loss

   -- Generate errors and losses
   local epoch = torch.linspace(1,#main.record,#main.record):mul(config.main.epoches)
   local train_error = torch.zeros(#main.record)
   local val_error = torch.zeros(#main.record)
   local train_loss = torch.zeros(#main.record)
   local val_loss = torch.zeros(#main.record)
   for i = 1,#main.record do
      train_error[i] = main.record[i].train_error
      val_error[i] = main.record[i].val_error
      train_loss[i] = main.record[i].train_loss
      val_loss[i] = main.record[i].val_loss
   end

   -- Do the plot
   gnuplot.figure(figure_error)
   gnuplot.plot({"Train",epoch,train_error},{"Validate",epoch,val_error})
   gnuplot.title("Training and validating error")
   gnuplot.plotflush()
   gnuplot.figure(figure_loss)
   gnuplot.plot({"Train",epoch,train_loss},{"Validate",epoch,val_loss})
   gnuplot.title("Training and validating loss")
   gnuplot.plotflush()
end

-- Draw the visualization
function main.draw()
   main.mui:drawSequential(main.model.sequential)
end

-- Save a record
function main.save()
   -- Record necessary configurations
   config.train.epoch = main.train.epoch

   -- Make the save
   local time = os.time()
   torch.save(paths.concat(config.main.save,"main_"..(main.train.epoch-1).."_"..time..".t7b"),
	      {config = config, record = main.record, momentum = main.train.old_grads:double()})
   torch.save(paths.concat(config.main.save,"sequential_"..(main.train.epoch-1).."_"..time..".t7b"),
	      main.model:clearSequential(main.model:makeCleanSequential(main.model.sequential)))
   main.eps_error = main.eps_error or gnuplot.epsfigure(paths.concat(config.main.save,"figure_error.eps"))
   main.eps_loss = main.eps_loss or gnuplot.epsfigure(paths.concat(config.main.save,"figure_loss.eps"))
   main.show(main.eps_error,main.eps_loss)
   local ret = pcall(function() main.mui.win:save(paths.concat(config.main.save,"sequential_"..(main.train.epoch-1).."_"..time..".png")) end)
   if not ret then print("Warning: saving the model image failed") end
   collectgarbage()
end

-- The training logging function
function main.trainlog(train)
   if config.main.collectgarbage and math.fmod(train.epoch-1,config.main.collectgarbage) == 0 then
      print("Collecting garbage at epoch = "..(train.epoch-1))
      collectgarbage()
   end

   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      local msg = ""
      
      if config.main.details then
	 msg = msg.."epo: "..(train.epoch-1)..
	    ", rat: "..string.format("%.2e",train.rate)..
	    ", err: "..string.format("%.2e",train.error)..
	    ", obj: "..string.format("%.2e",train.objective)..
	    ", dat: "..string.format("%.2e",train.time.data)..
	    ", fpp: "..string.format("%.2e",train.time.forward)..
	    ", bpp: "..string.format("%.2e",train.time.backward)..
	    ", upd: "..string.format("%.2e",train.time.update)
      end
      
      if config.main.debug then
	 msg = msg..", bmn: "..string.format("%.2e",train.batch:mean())..
	    ", bsd: "..string.format("%.2e",train.batch:std())..
	    ", bmi: "..string.format("%.2e",train.batch:min())..
	    ", bmx: "..string.format("%.2e",train.batch:max())..
	    ", pmn: "..string.format("%.2e",train.params:mean())..
	    ", psd: "..string.format("%.2e",train.params:std())..
	    ", pmi: "..string.format("%.2e",train.params:min())..
	    ", pmx: "..string.format("%.2e",train.params:max())..
	    ", gmn: "..string.format("%.2e",train.grads:mean())..
	    ", gsd: "..string.format("%.2e",train.grads:std())..
	    ", gmi: "..string.format("%.2e",train.grads:min())..
	    ", gmx: "..string.format("%.2e",train.grads:max())..
	    ", omn: "..string.format("%.2e",train.old_grads:mean())..
	    ", osd: "..string.format("%.2e",train.old_grads:std())..
	    ", omi: "..string.format("%.2e",train.old_grads:min())..
	    ", omx: "..string.format("%.2e",train.old_grads:max())
	 main.draw()
      end
      
      if config.main.details or config.main.debug then
	 print(msg)
      end

      main.clock.log = os.time()
   end
end

function main.testlog(test)
   if config.main.collectgarbage and math.fmod(test.n,config.train_data.batch_size*config.main.collectgarbage) == 0 then
      print("Collecting garbage at n = "..test.n)
      collectgarbage()
   end
   if not config.main.details then return end
   if (os.time() - main.clock.log) >= (config.main.logtime or 1) then
      print("n: "..test.n..
	       ", e: "..string.format("%.2e",test.e)..
	       ", l: "..string.format("%.2e",test.l)..
	       ", err: "..string.format("%.2e",test.err)..
	       ", obj: "..string.format("%.2e",test.objective)..
	       ", dat: "..string.format("%.2e",test.time.data)..
	       ", fpp: "..string.format("%.2e",test.time.forward)..
	       ", acc: "..string.format("%.2e",test.time.accumulate))
      main.clock.log = os.time()
   end
end

-- Utility function: find files with the specific 'ls' pattern
function main.findFiles(pattern)
   require("sys")
   local cmd = "ls "..pattern
   local str = sys.execute(cmd)
   local files = {}
   for file in str:gmatch("[^\n]+") do
      files[#files+1] = file
   end
   return files
end

-- Execute the main program
main.main()
