--[[ The schollable UI
By Xiang Zhang @ New York University
--]]

local Scroll = torch.class("Scroll")

-- Initialize a scroll interface
-- width: (optional) the pixel width of the scollable area. Default is 800.
-- title: (optional) title for the window
function Scroll:__init(width,title)
   require("qtuiloader")
   require("qtwidget")
   require("qttorch")

   self.file = "scroll.ui"
   self.win = qtuiloader.load(self.file)
   self.frame = self.win.frame
   self.painter = qt.QtLuaPainter(self.frame)
   self.width = width
   self.height = 0
   self.fontSize = 15
   self.x = 0
   self.y = 0
   self.border = 1

   self:resize(self.width, self.height)
   self:setFontSize(self.fontSize)
   if title then 
      self:setTitle(title)
   end
   self:show()
end

-- Resize the window to designated width and height
function Scroll:resize(width,height)
   self.width = width or self.width
   self.height = height or self.height

   self.frame.size = qt.QSize{width = self.width,height = self.height}
end

-- Set the text width
function Scroll:setFontSize(size)
   self.painter:setfontsize(size or 15)
   self.fontSize = size
end

-- Set border width
function Scroll:setBorder(width)
   self.border = width
end

-- Draw text
function Scroll:drawText(text)
   -- Drawing text must happen on a new line
   if self.x ~= 0 then
      self.x = 0
      self.y = self.height
   end

   -- Determine height and resize if necessary
   if self.height < self.y+self.fontSize+1 then
      self:resize(self.width,self.y+self.fontSize+1+self.border)
   end

   -- Draw the yellow main text
   self.painter:gbegin()
   self.painter:moveto(self.x,self.y+self.fontSize-1)
   self.painter:setcolor(1,1,0,1)
   self.painter:show(text)
   self.painter:stroke()
   self.painter:gend()

   -- Draw the black shadow text
   self.painter:gbegin()
   self.painter:moveto(self.x,self.y+self.fontSize+1-1)
   self.painter:setcolor(0,0,0,1)
   self.painter:show(text)
   self.painter:stroke()
   self.painter:gend()
   
   -- Move the cursor to next line
   self.x = 0
   if self.height < self.y+self.fontSize+1+self.border then
      self:resize(self.width,self.y+self.fontSize+1+self.border)
   end
   self.y = self.height
end

-- Draw image
function Scroll:drawImage(im,scale)
   -- Get the image height and width
   local scale = scale or 1
   local height,width
   if im:dim() == 2 then
      height = im:size(1)*scale
      width = im:size(2)*scale
   elseif im:dim() == 3 then
      height = im:size(2)*scale
      width = im:size(3)*scale
   else
      error("Image must be 2-dim or 3-dim data")
   end

   -- Determine whether a new line is needed
   if self.x ~=0 and self.x + width > self.width then
      self.x = 0
      self.y = self.height
   end

   -- Determine whether need to resize the document area
   if self.y + height > self.height then
      self:resize(self.width,self.y+height+self.border)
   end

   -- Draw the image
   self.painter:gbegin()
   self.painter:image(self.x,self.y,width,height,qt.QImage.fromTensor(im))
   self.painter:stroke()
   self.painter:gend()

   -- Move the cursor
   self.x = self.x + width + self.border
end

-- Draw a new line
function Scroll:drawEndOfLine()
   self.x = 0
   self.y = self.height
end

-- Show the window
function Scroll:show()
   self.win:show()
end

-- Hide the window
function Scroll:hide()
   self.win:hide()
end

-- Save to file
function Scroll:save(file)
   self.painter:write(file)
end

-- Set window title
function Scroll:setTitle(title)
   self.win:setWindowTitle(title)
end

-- Reset the drawing area
function Scroll:clear()
   self:resize(self.width,0)
   self.x = 0
   self.y = 0
end
