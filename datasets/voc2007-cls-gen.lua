
local sys = require 'sys'
local ffi = require 'ffi'
local csvigo = require 'csvigo'
local paths = require 'paths'

local M = {}

M.nClasses = 20

local function findImages(file, opt)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   local m = csvigo.load({path = file, mode = "large"})

   local maxLength = -1
   local imagePaths = {}
   local imageClasses = {}

   local pathImages = opt.data .. "/JPEGImages/"

   -- Generate a list of all the images and their class
   for t=2, #m  do
     local path = m[t][1] .. '.jpg'
     local labels = torch.Tensor(M.nClasses)
     for c = 1, M.nClasses do labels[c] = m[t][c+1] end

      table.insert(imagePaths, path)
      table.insert(imageClasses, labels)

      maxLength = math.max(maxLength, #path + 1)
   end

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   local imageClass = torch.zeros(nImages, M.nClasses)
   for i=1,#imageClasses do
     imageClass[i] = imageClasses[i]
   end
   return imagePath, imageClass
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   print(" | finding all training images")
   local filename = 'data/datasets/voc2007_cls_trainval.csv'
   paths.mkdir('data/datasets')
   if not paths.filep(filename) then
       local url = 'http://webia.lip6.fr/~durandt/data/dataset/voc2007_cls_trainval.csv'
       os.execute('wget ' .. url .. ' -O ' .. filename)
   end
   local trainImagePath, trainImageClass = findImages(filename, opt)

   print(" | finding all testing images")
   local filename = 'data/datasets/voc2007_cls_test.csv'
   if not paths.filep(filename) then
       local url = 'http://webia.lip6.fr/~durandt/data/dataset/voc2007_cls_test.csv'
       os.execute('wget ' .. url .. ' -O ' .. filename)
   end
   local testImagePath, testImageClass = findImages(filename, opt)

   local info = {
      basedir = opt.data,
      nClasses = M.nClasses,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
      },
      test = {
         imagePath = testImagePath,
         imageClass = testImageClass,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
